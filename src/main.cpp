// Gameplay Recorder - Geode mod for Geometry Dash (Android)
//
// Video : copies the finished frame into a texture, draws it onto the input
//         surface of a hardware H.264 encoder (no glReadPixels = low lag).
// Audio : an FMOD DSP on the master channel group grabs the game's sound
//         and feeds a hardware AAC encoder (no mic / permissions needed).
// Smooth: by default the drawing/encoding hand-off runs on its own thread with
//         a shared GL context, so the game thread only does a quick GPU copy and
//         never waits (frames are dropped from the video instead of lagging).
// Output: MP4 in the mod's save folder (<save dir>/recordings).
//
// Start/stop with the REC button on the pause menu.
//
// NOTE: written without being able to compile it. If the build fails,
// copy the error log and fix the line it points at.

#include <Geode/Geode.hpp>
#include <Geode/modify/CCDirector.hpp>
#include <Geode/modify/PauseLayer.hpp>

#include <media/NdkMediaCodec.h>
#include <media/NdkMediaFormat.h>
#include <media/NdkMediaMuxer.h>
#include <android/native_window.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <thread>
#include <vector>

using namespace geode::prelude;

#ifndef EGL_RECORDABLE_ANDROID
#define EGL_RECORDABLE_ANDROID 0x3142
#endif

static int64_t nowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------
// Lock-free single-producer / single-consumer ring buffer for audio samples
// ---------------------------------------------------------------------------
struct Ring {
    static constexpr size_t N = 1 << 20; // samples (int16), power of two
    std::vector<int16_t> buf = std::vector<int16_t>(N);
    std::atomic<size_t> head{0}, tail{0};

    size_t write(const int16_t* d, size_t n) {
        size_t h = head.load(std::memory_order_relaxed);
        size_t t = tail.load(std::memory_order_acquire);
        size_t freeS = N - (h - t);
        if (n > freeS) n = freeS;
        for (size_t i = 0; i < n; i++) buf[(h + i) & (N - 1)] = d[i];
        head.store(h + n, std::memory_order_release);
        return n;
    }
    size_t read(int16_t* d, size_t n) {
        size_t h = head.load(std::memory_order_acquire);
        size_t t = tail.load(std::memory_order_relaxed);
        size_t avail = h - t;
        if (n > avail) n = avail;
        for (size_t i = 0; i < n; i++) d[i] = buf[(t + i) & (N - 1)];
        tail.store(t + n, std::memory_order_release);
        return n;
    }
    size_t size() const {
        return head.load(std::memory_order_acquire) - tail.load(std::memory_order_acquire);
    }
    void reset() { head.store(0); tail.store(0); }
};

static FMOD_RESULT dspRead(FMOD_DSP_STATE*, float* in, float* out,
                           unsigned int length, int inch, int* outch);

struct PendingSample {
    int track;
    std::vector<uint8_t> data;
    AMediaCodecBufferInfo info;
};

// ---------------------------------------------------------------------------
// The recorder
// ---------------------------------------------------------------------------
struct Recorder {
    // settings
    int fps = 60, w = 0, h = 0, sw = 0, sh = 0, sampleRate = 48000;
    int32_t bitrate = 12000000;
    bool wantAudio = true;

    // state
    std::atomic<bool> on{false}, audioOn{false}, stopping{false};
    std::atomic<int64_t> stopAtNs{0};
    int64_t t0 = 0, nextFrameNs = 0;
    std::string outPath;

    // EGL / GL (render thread only)
    EGLDisplay dpy = EGL_NO_DISPLAY;
    EGLContext ctx = EGL_NO_CONTEXT;
    EGLSurface surf = EGL_NO_SURFACE;
    ANativeWindow* win = nullptr;
    GLuint prog = 0, tex = 0;
    GLint uTex = -1;
    PFNEGLPRESENTATIONTIMEANDROIDPROC presentTime = nullptr;

    // smooth (threaded) capture
    static constexpr int kSlots = 3;
    struct Slot {
        GLuint tex = 0;
        EGLSyncKHR fence = EGL_NO_SYNC_KHR;
        int64_t pts = 0;
        std::atomic<int> state{0}; // 0 free, 1 ready to draw
    };
    Slot slots[kSlots];
    int writeIdx = 0;
    bool async = false;
    EGLContext rctx = EGL_NO_CONTEXT;
    std::thread rthread;
    std::mutex rmu;
    std::condition_variable rcv;
    std::atomic<bool> rquit{false}, rFailed{false};
    PFNEGLCREATESYNCKHRPROC eglCreateSync_ = nullptr;
    PFNEGLCLIENTWAITSYNCKHRPROC eglClientWait_ = nullptr;
    PFNEGLDESTROYSYNCKHRPROC eglDestroySync_ = nullptr;

    // codecs / muxer
    AMediaCodec* venc = nullptr;
    AMediaCodec* aenc = nullptr;
    AMediaMuxer* mux = nullptr;
    int fd = -1;
    int vTrack = -1, aTrack = -1;
    bool muxStarted = false;
    std::vector<PendingSample> pending;
    std::thread worker;

    // audio
    Ring ring;
    FMOD::DSP* dsp = nullptr;

    // -----------------------------------------------------------------------
    bool fail(const char* msg) {
        log::error("Recorder: {}", msg);
        Notification::create(std::string("Recording failed: ") + msg,
                             NotificationIcon::Error)->show();
        teardown(true);
        return false;
    }

    // ---------------------------- GL helpers -------------------------------
    static GLuint compile(GLenum type, const char* src) {
        GLuint s = glCreateShader(type);
        glShaderSource(s, 1, &src, nullptr);
        glCompileShader(s);
        GLint ok = 0;
        glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
        if (!ok) { glDeleteShader(s); return 0; }
        return s;
    }

    static GLuint buildProgram(GLint& uTexOut) {
        const char* vs =
            "attribute vec2 aPos; attribute vec2 aUV; varying vec2 vUV;"
            "void main(){ gl_Position = vec4(aPos, 0.0, 1.0); vUV = aUV; }";
        const char* fs =
            "precision mediump float; varying vec2 vUV; uniform sampler2D uTex;"
            "void main(){ gl_FragColor = vec4(texture2D(uTex, vUV).rgb, 1.0); }";
        GLuint v = compile(GL_VERTEX_SHADER, vs);
        GLuint f = compile(GL_FRAGMENT_SHADER, fs);
        if (!v || !f) return 0;
        GLuint p = glCreateProgram();
        glAttachShader(p, v);
        glAttachShader(p, f);
        glBindAttribLocation(p, 0, "aPos");
        glBindAttribLocation(p, 1, "aUV");
        glLinkProgram(p);
        glDeleteShader(v);
        glDeleteShader(f);
        GLint ok = 0;
        glGetProgramiv(p, GL_LINK_STATUS, &ok);
        if (!ok) { glDeleteProgram(p); return 0; }
        uTexOut = glGetUniformLocation(p, "uTex");
        return p;
    }

    void allocTexture(GLuint& t) {
        glGenTextures(1, &t);
        glBindTexture(GL_TEXTURE_2D, t);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, sw, sh, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    }

    bool initGL(bool useAsync) {
        if (!useAsync) {
            prog = buildProgram(uTex);
            if (!prog) return false;
        }
        GLint oldTex = 0, oldAct = 0;
        glGetIntegerv(GL_ACTIVE_TEXTURE, &oldAct);
        glActiveTexture(GL_TEXTURE0);
        glGetIntegerv(GL_TEXTURE_BINDING_2D, &oldTex);
        if (useAsync) {
            for (auto& sl : slots) {
                allocTexture(sl.tex);
                sl.state = 0;
                sl.fence = EGL_NO_SYNC_KHR;
            }
        } else {
            allocTexture(tex);
        }
        glBindTexture(GL_TEXTURE_2D, oldTex);
        glActiveTexture(oldAct);
        return true;
    }

    EGLSurface makeSurface(EGLConfig& cfg) {
        EGLint cfgId = 0;
        eglQueryContext(dpy, ctx, EGL_CONFIG_ID, &cfgId);
        EGLint n = 0;
        EGLint a1[] = {EGL_CONFIG_ID, cfgId, EGL_NONE};
        if (eglChooseConfig(dpy, a1, &cfg, 1, &n) && n > 0) {
            EGLSurface s = eglCreateWindowSurface(dpy, cfg, win, nullptr);
            if (s != EGL_NO_SURFACE) return s;
        }
        EGLint a2[] = {
            EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
            EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
            EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8,
            EGL_RECORDABLE_ANDROID, 1,
            EGL_NONE};
        if (eglChooseConfig(dpy, a2, &cfg, 1, &n) && n > 0)
            return eglCreateWindowSurface(dpy, cfg, win, nullptr);
        return EGL_NO_SURFACE;
    }

    // ------------------------------ start ----------------------------------
    bool start() {
        if (on.load()) return false;

        auto* mod = Mod::get();
        auto q = mod->getSettingValue<std::string>("quality");
        int targetH = q == "1080p" ? 1080 : (q == "480p" ? 480 : 720);
        fps = (int)mod->getSettingValue<int64_t>("fps");
        bitrate = (int32_t)(mod->getSettingValue<int64_t>("bitrate") * 1000000);
        wantAudio = mod->getSettingValue<bool>("record-audio");
        bool smooth = mod->getSettingValue<bool>("smooth-capture");

        dpy = eglGetCurrentDisplay();
        ctx = eglGetCurrentContext();
        EGLSurface gameSurf = eglGetCurrentSurface(EGL_DRAW);
        if (ctx == EGL_NO_CONTEXT || gameSurf == EGL_NO_SURFACE)
            return fail("no GL context");
        EGLint ew = 0, eh = 0;
        eglQuerySurface(dpy, gameSurf, EGL_WIDTH, &ew);
        eglQuerySurface(dpy, gameSurf, EGL_HEIGHT, &eh);
        sw = ew;
        sh = eh;
        if (sw <= 0 || sh <= 0) return fail("bad screen size");

        h = std::min(targetH, sh) & ~1;
        w = ((int)(h * (double)sw / (double)sh + 0.5)) & ~1;

        // ---- output file + muxer
        auto dir = Mod::get()->getSaveDir() / "recordings";
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        char name[64];
        std::time_t tt = std::time(nullptr);
        std::tm tmv;
        localtime_r(&tt, &tmv);
        std::strftime(name, sizeof(name), "GDRec_%Y%m%d_%H%M%S.mp4", &tmv);
        outPath = (dir / name).string();
        fd = open(outPath.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0666);
        if (fd < 0) return fail("cannot create file");
        mux = AMediaMuxer_new(fd, AMEDIAMUXER_OUTPUT_FORMAT_MPEG_4);
        if (!mux) return fail("cannot create muxer");

        // ---- video encoder
        AMediaFormat* f = AMediaFormat_new();
        AMediaFormat_setString(f, "mime", "video/avc");
        AMediaFormat_setInt32(f, "width", w);
        AMediaFormat_setInt32(f, "height", h);
        AMediaFormat_setInt32(f, "bitrate", bitrate);
        AMediaFormat_setInt32(f, "frame-rate", fps);
        AMediaFormat_setInt32(f, "i-frame-interval", 1);
        AMediaFormat_setInt32(f, "color-format", 0x7F000789); // COLOR_FormatSurface
        AMediaFormat_setInt32(f, "priority", 0);              // real-time
        venc = AMediaCodec_createEncoderByType("video/avc");
        if (!venc) { AMediaFormat_delete(f); return fail("no H.264 encoder"); }
        media_status_t st = AMediaCodec_configure(
            venc, f, nullptr, nullptr, AMEDIACODEC_CONFIGURE_FLAG_ENCODE);
        AMediaFormat_delete(f);
        if (st != AMEDIA_OK) return fail("encoder config rejected (try lower quality/fps)");

        // createInputSurface needs Android 8+, so look it up at runtime
        using CreateSurfaceFn = media_status_t (*)(AMediaCodec*, ANativeWindow**);
        auto createSurface = (CreateSurfaceFn)dlsym(RTLD_DEFAULT, "AMediaCodec_createInputSurface");
        if (!createSurface) return fail("Android 8+ required");
        if (createSurface(venc, &win) != AMEDIA_OK || !win) return fail("no encoder surface");
        if (AMediaCodec_start(venc) != AMEDIA_OK) return fail("encoder start failed");

        // ---- EGL surface + GL objects
        EGLConfig cfg = nullptr;
        surf = makeSurface(cfg);
        if (surf == EGL_NO_SURFACE) return fail("cannot make EGL surface");
        presentTime = (PFNEGLPRESENTATIONTIMEANDROIDPROC)eglGetProcAddress("eglPresentationTimeANDROID");
        eglCreateSync_ = (PFNEGLCREATESYNCKHRPROC)eglGetProcAddress("eglCreateSyncKHR");
        eglClientWait_ = (PFNEGLCLIENTWAITSYNCKHRPROC)eglGetProcAddress("eglClientWaitSyncKHR");
        eglDestroySync_ = (PFNEGLDESTROYSYNCKHRPROC)eglGetProcAddress("eglDestroySyncKHR");

        async = false;
        if (smooth) {
            EGLint cattr[] = {EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE};
            rctx = eglCreateContext(dpy, cfg, ctx, cattr);
            if (rctx != EGL_NO_CONTEXT) async = true;
            else log::warn("Recorder: shared context failed, using simple capture");
        }
        if (!initGL(async)) return fail("GL init failed");
        writeIdx = 0;
        rquit = false;
        rFailed = false;

        // ---- audio (optional; falls back to video only)
        if (wantAudio && !setupAudio()) {
            log::warn("Audio setup failed, recording video only");
            Notification::create("Audio capture unavailable, recording video only",
                                 NotificationIcon::Warning)->show();
            wantAudio = false;
        }

        ring.reset();
        t0 = nowNs();
        nextFrameNs = t0;
        stopping = false;
        on = true;
        audioOn = wantAudio;
        worker = std::thread([this] { workerLoop(); });
        if (async) rthread = std::thread([this] { renderLoop(); });

        Notification::create("Recording started", NotificationIcon::Success)->show();
        return true;
    }

    bool setupAudio() {
        auto* fm = FMODAudioEngine::sharedEngine();
        FMOD::System* sys = fm ? fm->m_system : nullptr;
        if (!sys) return false;
        int rate = 0;
        sys->getSoftwareFormat(&rate, nullptr, nullptr);
        sampleRate = rate > 0 ? rate : 48000;

        AMediaFormat* f = AMediaFormat_new();
        AMediaFormat_setString(f, "mime", "audio/mp4a-latm");
        AMediaFormat_setInt32(f, "sample-rate", sampleRate);
        AMediaFormat_setInt32(f, "channel-count", 2);
        AMediaFormat_setInt32(f, "bitrate", 192000);
        AMediaFormat_setInt32(f, "aac-profile", 2); // AAC-LC
        AMediaFormat_setInt32(f, "max-input-size", 16384);
        aenc = AMediaCodec_createEncoderByType("audio/mp4a-latm");
        if (!aenc) { AMediaFormat_delete(f); return false; }
        media_status_t st = AMediaCodec_configure(
            aenc, f, nullptr, nullptr, AMEDIACODEC_CONFIGURE_FLAG_ENCODE);
        AMediaFormat_delete(f);
        if (st != AMEDIA_OK) return false;
        if (AMediaCodec_start(aenc) != AMEDIA_OK) return false;

        FMOD::ChannelGroup* master = nullptr;
        if (sys->getMasterChannelGroup(&master) != FMOD_OK || !master) return false;
        FMOD_DSP_DESCRIPTION d;
        std::memset(&d, 0, sizeof(d));
        d.pluginsdkversion = FMOD_PLUGIN_SDK_VERSION;
        std::strncpy(d.name, "GDRecorder", sizeof(d.name) - 1);
        d.version = 1;
        d.numinputbuffers = 1;
        d.numoutputbuffers = 1;
        d.read = dspRead;
        if (sys->createDSP(&d, &dsp) != FMOD_OK || !dsp) return false;
        if (master->addDSP(FMOD_CHANNELCONTROL_DSP_TAIL, dsp) != FMOD_OK) {
            dsp->release();
            dsp = nullptr;
            return false;
        }
        return true;
    }

    void detachDsp() {
        if (!dsp) return;
        auto* fm = FMODAudioEngine::sharedEngine();
        FMOD::ChannelGroup* master = nullptr;
        if (fm && fm->m_system && fm->m_system->getMasterChannelGroup(&master) == FMOD_OK && master)
            master->removeDSP(dsp);
        dsp->release();
        dsp = nullptr;
    }

    // ------------------------------ stop -----------------------------------
    void stop(bool glValid = true) {
        if (!on.load()) return;
        on = false;
        audioOn = false;
        detachDsp();
        if (rthread.joinable()) {
            rquit = true;
            rcv.notify_all();
            rthread.join();
        }
        stopAtNs = nowNs() + 3000000000LL;
        stopping = true;
        if (venc) {
            using SignalEosFn = media_status_t (*)(AMediaCodec*);
            auto signalEos = (SignalEosFn)dlsym(RTLD_DEFAULT, "AMediaCodec_signalEndOfInputStream");
            if (signalEos) signalEos(venc);
        }
        if (worker.joinable()) worker.join();
        std::string path = outPath;
        teardown(glValid);
        log::info("Recording saved: {}", path);
        Notification::create("Saved: " + path, NotificationIcon::Success, 5.f)->show();
    }

    void teardown(bool glValid) {
        detachDsp();
        if (aenc) { AMediaCodec_stop(aenc); AMediaCodec_delete(aenc); aenc = nullptr; }
        if (venc) { AMediaCodec_stop(venc); AMediaCodec_delete(venc); venc = nullptr; }
        if (mux) {
            if (muxStarted) AMediaMuxer_stop(mux);
            AMediaMuxer_delete(mux);
            mux = nullptr;
        }
        if (fd >= 0) { close(fd); fd = -1; }
        for (auto& sl : slots) {
            if (sl.fence != EGL_NO_SYNC_KHR && eglDestroySync_) eglDestroySync_(dpy, sl.fence);
            sl.fence = EGL_NO_SYNC_KHR;
        }
        if (rctx != EGL_NO_CONTEXT) { eglDestroyContext(dpy, rctx); rctx = EGL_NO_CONTEXT; }
        if (surf != EGL_NO_SURFACE) { eglDestroySurface(dpy, surf); surf = EGL_NO_SURFACE; }
        if (win) { ANativeWindow_release(win); win = nullptr; }
        if (glValid) {
            if (prog) glDeleteProgram(prog);
            if (tex) glDeleteTextures(1, &tex);
            for (auto& sl : slots)
                if (sl.tex) glDeleteTextures(1, &sl.tex);
        }
        prog = 0;
        tex = 0;
        for (auto& sl : slots) { sl.tex = 0; sl.state = 0; }
        async = false;
        vTrack = aTrack = -1;
        muxStarted = false;
        pending.clear();
    }

    // --------------------------- worker thread -----------------------------
    void maybeStartMux() {
        if (muxStarted || vTrack < 0 || (wantAudio && aTrack < 0)) return;
        if (AMediaMuxer_start(mux) != AMEDIA_OK) {
            log::error("Muxer failed to start");
            return;
        }
        muxStarted = true;
        for (auto& p : pending)
            AMediaMuxer_writeSampleData(mux, p.track, p.data.data(), &p.info);
        pending.clear();
    }

    void writeSample(int track, const uint8_t* data, const AMediaCodecBufferInfo& info) {
        if (muxStarted) {
            AMediaCodecBufferInfo i = info;
            i.offset = 0;
            AMediaMuxer_writeSampleData(mux, track, data, &i);
        } else {
            PendingSample p;
            p.track = track;
            p.data.assign(data, data + info.size);
            p.info = info;
            p.info.offset = 0;
            pending.push_back(std::move(p));
        }
    }

    bool drain(AMediaCodec* c, int& track, bool& eos) {
        bool any = false;
        for (int guard = 0; guard < 16 && !eos; guard++) {
            AMediaCodecBufferInfo info;
            ssize_t idx = AMediaCodec_dequeueOutputBuffer(c, &info, 0);
            if (idx == AMEDIACODEC_INFO_TRY_AGAIN_LATER) break;
            any = true;
            if (idx == AMEDIACODEC_INFO_OUTPUT_FORMAT_CHANGED) {
                AMediaFormat* fmt = AMediaCodec_getOutputFormat(c);
                track = AMediaMuxer_addTrack(mux, fmt);
                AMediaFormat_delete(fmt);
                maybeStartMux();
            } else if (idx >= 0) {
                size_t sz = 0;
                uint8_t* data = AMediaCodec_getOutputBuffer(c, idx, &sz);
                if (data && info.size > 0 && track >= 0 &&
                    !(info.flags & AMEDIACODEC_BUFFER_FLAG_CODEC_CONFIG)) {
                    writeSample(track, data + info.offset, info);
                }
                if (info.flags & AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM) eos = true;
                AMediaCodec_releaseOutputBuffer(c, idx, false);
            }
        }
        return any;
    }

    bool feedAudio(int64_t& frames, bool& done) {
        if (stopping.load()) {
            ssize_t i = AMediaCodec_dequeueInputBuffer(aenc, 0);
            if (i < 0) return false;
            AMediaCodec_queueInputBuffer(aenc, i, 0, 0,
                                         frames * 1000000LL / sampleRate,
                                         AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM);
            done = true;
            return true;
        }
        size_t n = 0;
        bool silence = false;
        size_t avail = ring.size();
        if (avail > 0) {
            n = std::min<size_t>(avail, 4096);
        } else {
            // game is silent / DSP idle: keep the audio timeline moving
            int64_t expected = (nowNs() - t0) * (int64_t)sampleRate / 1000000000LL;
            if (frames + 4800 < expected) { n = 2048; silence = true; }
        }
        if (n == 0) return false;
        ssize_t i = AMediaCodec_dequeueInputBuffer(aenc, 0);
        if (i < 0) return false;
        size_t cap = 0;
        uint8_t* buf = AMediaCodec_getInputBuffer(aenc, i, &cap);
        if (!buf) return false;
        n = std::min(n, cap / 2);
        n &= ~(size_t)1;
        if (silence) std::memset(buf, 0, n * 2);
        else n = ring.read(reinterpret_cast<int16_t*>(buf), n);
        AMediaCodec_queueInputBuffer(aenc, i, 0, n * 2,
                                     frames * 1000000LL / sampleRate, 0);
        frames += (int64_t)(n / 2);
        return true;
    }

    void workerLoop() {
        bool vEos = false;
        bool aEos = !wantAudio;
        bool aInDone = !wantAudio;
        int64_t aFrames = 0;
        while (true) {
            bool busy = false;
            busy |= drain(venc, vTrack, vEos);
            if (wantAudio) {
                if (!aInDone) busy |= feedAudio(aFrames, aInDone);
                busy |= drain(aenc, aTrack, aEos);
            }
            if (vEos && aEos) break;
            if (stopping.load() && nowNs() > stopAtNs.load()) {
                log::warn("Recorder: timed out while finishing");
                break;
            }
            if (!busy) std::this_thread::sleep_for(std::chrono::milliseconds(3));
        }
    }

    // ------------------------ threaded render loop --------------------------
    void renderLoop() {
        if (!eglMakeCurrent(dpy, surf, surf, rctx)) { rFailed = true; return; }
        GLint u = -1;
        GLuint p = buildProgram(u);
        if (!p) {
            rFailed = true;
            eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
            return;
        }
        glViewport(0, 0, w, h);
        glDisable(GL_BLEND);
        glDisable(GL_DEPTH_TEST);
        glUseProgram(p);
        glUniform1i(u, 0);
        glActiveTexture(GL_TEXTURE0);
        static const GLfloat pos[] = {-1.f, -1.f, 1.f, -1.f, -1.f, 1.f, 1.f, 1.f};
        static const GLfloat uvs[] = {0.f, 0.f, 1.f, 0.f, 0.f, 1.f, 1.f, 1.f};
        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, pos);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, uvs);

        int readIdx = 0;
        while (true) {
            Slot& s = slots[readIdx];
            if (s.state.load(std::memory_order_acquire) != 1) {
                if (rquit.load()) break;
                std::unique_lock<std::mutex> lk(rmu);
                rcv.wait_for(lk, std::chrono::milliseconds(5), [&] {
                    return s.state.load(std::memory_order_acquire) == 1 || rquit.load();
                });
                continue;
            }
            if (s.fence != EGL_NO_SYNC_KHR) {
                if (eglClientWait_)
                    eglClientWait_(dpy, s.fence, EGL_SYNC_FLUSH_COMMANDS_BIT_KHR, 100000000ULL);
                if (eglDestroySync_) eglDestroySync_(dpy, s.fence);
                s.fence = EGL_NO_SYNC_KHR;
            }
            glBindTexture(GL_TEXTURE_2D, s.tex);
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
            if (presentTime) presentTime(dpy, surf, s.pts);
            eglSwapBuffers(dpy, surf); // may block, but only this thread
            s.state.store(0, std::memory_order_release);
            readIdx = (readIdx + 1) % kSlots;
        }
        glDeleteProgram(p);
        eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    }

    // ------------------------ per-frame capture (GL thread) -----------------
    void captureFrame() {
        if (!on.load()) return;

        int64_t now = nowNs();
        int64_t interval = 1000000000LL / fps;
        if (now + interval / 4 < nextFrameNs) return;
        nextFrameNs += interval;
        if (nextFrameNs < now) nextFrameNs = now + interval;

        if (eglGetCurrentContext() != ctx || eglGetCurrentDisplay() != dpy) {
            log::warn("GL context changed (app in background?), stopping recording");
            stop(false);
            return;
        }
        if (rFailed.load()) {
            Notification::create("Smooth capture failed. Turn it off in the mod settings.",
                                 NotificationIcon::Error)->show();
            stop(true);
            return;
        }
        EGLSurface drawS = eglGetCurrentSurface(EGL_DRAW);
        EGLSurface readS = eglGetCurrentSurface(EGL_READ);
        EGLint cw = 0, ch = 0;
        eglQuerySurface(dpy, drawS, EGL_WIDTH, &cw);
        eglQuerySurface(dpy, drawS, EGL_HEIGHT, &ch);
        if (cw <= 0 || ch <= 0) return;

        if (async) captureAsync(now, cw, ch);
        else captureSync(now, drawS, readS, cw, ch);
    }

    // Smooth path: only a GPU copy + fence on the game thread.
    void captureAsync(int64_t now, EGLint cw, EGLint ch) {
        if (cw != sw || ch != sh) {
            Notification::create("Screen size changed, recording stopped",
                                 NotificationIcon::Warning)->show();
            stop(true);
            return;
        }
        Slot& s = slots[writeIdx];
        if (s.state.load(std::memory_order_acquire) != 0) return; // encoder busy: drop frame, never wait

        GLint oFbo = 0, oTex = 0, oAct = 0;
        glGetIntegerv(GL_FRAMEBUFFER_BINDING, &oFbo);
        glGetIntegerv(GL_ACTIVE_TEXTURE, &oAct);
        glActiveTexture(GL_TEXTURE0);
        glGetIntegerv(GL_TEXTURE_BINDING_2D, &oTex);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glBindTexture(GL_TEXTURE_2D, s.tex);
        glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, sw, sh);
        if (eglCreateSync_) s.fence = eglCreateSync_(dpy, EGL_SYNC_FENCE_KHR, nullptr);
        else glFinish();
        glFlush();

        glBindTexture(GL_TEXTURE_2D, oTex);
        glActiveTexture(oAct);
        glBindFramebuffer(GL_FRAMEBUFFER, oFbo);

        s.pts = now - t0;
        s.state.store(1, std::memory_order_release);
        writeIdx = (writeIdx + 1) % kSlots;
        rcv.notify_one();
    }

    // Simple path (fallback): draws to the encoder on the game thread.
    void captureSync(int64_t now, EGLSurface drawS, EGLSurface readS, EGLint cw, EGLint ch) {
        // ---- save the game's GL state
        GLint oFbo = 0, oProg = 0, oTex = 0, oBuf = 0, oAct = 0, oVp[4] = {0, 0, 0, 0}, oAttr[3] = {0, 0, 0};
        GLboolean oMask[4] = {1, 1, 1, 1};
        glGetIntegerv(GL_FRAMEBUFFER_BINDING, &oFbo);
        glGetIntegerv(GL_CURRENT_PROGRAM, &oProg);
        glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &oBuf);
        glGetIntegerv(GL_ACTIVE_TEXTURE, &oAct);
        glGetIntegerv(GL_VIEWPORT, oVp);
        glGetBooleanv(GL_COLOR_WRITEMASK, oMask);
        glActiveTexture(GL_TEXTURE0);
        glGetIntegerv(GL_TEXTURE_BINDING_2D, &oTex);
        for (int i = 0; i < 3; i++)
            glGetVertexAttribiv(i, GL_VERTEX_ATTRIB_ARRAY_ENABLED, &oAttr[i]);
        GLboolean oBlend = glIsEnabled(GL_BLEND);
        GLboolean oDepth = glIsEnabled(GL_DEPTH_TEST);
        GLboolean oScis = glIsEnabled(GL_SCISSOR_TEST);
        GLboolean oCull = glIsEnabled(GL_CULL_FACE);
        GLboolean oSten = glIsEnabled(GL_STENCIL_TEST);

        // ---- copy the finished frame into our texture
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glBindTexture(GL_TEXTURE_2D, tex);
        if (cw != sw || ch != sh) {
            sw = cw;
            sh = ch;
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, sw, sh, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
        }
        glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, sw, sh);

        // ---- draw it onto the encoder surface
        bool ok = eglMakeCurrent(dpy, surf, surf, ctx);
        if (ok) {
            glViewport(0, 0, w, h);
            glDisable(GL_BLEND);
            glDisable(GL_DEPTH_TEST);
            glDisable(GL_SCISSOR_TEST);
            glDisable(GL_CULL_FACE);
            glDisable(GL_STENCIL_TEST);
            glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, tex);
            glUseProgram(prog);
            glUniform1i(uTex, 0);
            static const GLfloat pos[] = {-1.f, -1.f, 1.f, -1.f, -1.f, 1.f, 1.f, 1.f};
            static const GLfloat uvs[] = {0.f, 0.f, 1.f, 0.f, 0.f, 1.f, 1.f, 1.f};
            glEnableVertexAttribArray(0);
            glEnableVertexAttribArray(1);
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, pos);
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, uvs);
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
            if (presentTime) presentTime(dpy, surf, now - t0);
            eglSwapBuffers(dpy, surf);
        }

        // ---- back to the game's surface, restore its GL state
        eglMakeCurrent(dpy, drawS, readS, ctx);
        for (int i = 0; i < 3; i++) {
            if (oAttr[i]) glEnableVertexAttribArray(i);
            else glDisableVertexAttribArray(i);
        }
        glUseProgram(oProg);
        glBindBuffer(GL_ARRAY_BUFFER, oBuf);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, oTex);
        glActiveTexture(oAct);
        glBindFramebuffer(GL_FRAMEBUFFER, oFbo);
        glViewport(oVp[0], oVp[1], oVp[2], oVp[3]);
        glColorMask(oMask[0], oMask[1], oMask[2], oMask[3]);
        if (oBlend) glEnable(GL_BLEND); else glDisable(GL_BLEND);
        if (oDepth) glEnable(GL_DEPTH_TEST); else glDisable(GL_DEPTH_TEST);
        if (oScis) glEnable(GL_SCISSOR_TEST); else glDisable(GL_SCISSOR_TEST);
        if (oCull) glEnable(GL_CULL_FACE); else glDisable(GL_CULL_FACE);
        if (oSten) glEnable(GL_STENCIL_TEST); else glDisable(GL_STENCIL_TEST);

        if (!ok) {
            log::error("Recorder: eglMakeCurrent on encoder surface failed");
            stop(true);
        }
    }
};

static Recorder g_rec;

// FMOD DSP callback: runs on FMOD's mixer thread. Passes audio through
// untouched and copies a stereo 16-bit version into the ring buffer.
static FMOD_RESULT dspRead(FMOD_DSP_STATE*, float* in, float* out,
                           unsigned int length, int inch, int* outch) {
    if (outch) *outch = inch;
    if (!out || inch <= 0) return FMOD_OK;
    if (!in) {
        std::memset(out, 0, sizeof(float) * length * (size_t)inch);
        return FMOD_OK;
    }
    if (out != in) std::memcpy(out, in, sizeof(float) * length * (size_t)inch);

    if (g_rec.audioOn.load(std::memory_order_relaxed)) {
        int16_t tmp[1024];
        unsigned int done = 0;
        while (done < length) {
            unsigned int n = std::min(512u, length - done);
            for (unsigned int i = 0; i < n; i++) {
                const float* fr = in + (size_t)(done + i) * (size_t)inch;
                float l = fr[0];
                float r = inch > 1 ? fr[1] : fr[0];
                l = std::clamp(l, -1.f, 1.f);
                r = std::clamp(r, -1.f, 1.f);
                tmp[i * 2] = (int16_t)(l * 32767.f);
                tmp[i * 2 + 1] = (int16_t)(r * 32767.f);
            }
            g_rec.ring.write(tmp, (size_t)n * 2);
            done += n;
        }
    }
    return FMOD_OK;
}

// ---------------------------------------------------------------------------
// Hooks
// ---------------------------------------------------------------------------

// After the game finishes drawing a frame (Android swaps the buffers later,
// in Java), grab it.
class $modify(RecDirector, CCDirector) {
    void drawScene() {
        CCDirector::drawScene();
        if (g_rec.on.load()) g_rec.captureFrame();
    }
};

// REC / Stop button on the pause menu.
class $modify(RecPauseLayer, PauseLayer) {
    void customSetup() {
        PauseLayer::customSetup();
        this->addRecButton();
    }

    void addRecButton() {
        if (auto old = this->getChildByID("rec-menu"_spr)) old->removeFromParent();

        auto win = CCDirector::get()->getWinSize();
        auto spr = ButtonSprite::create(g_rec.on.load() ? "Stop REC" : "REC");
        spr->setScale(0.7f);
        auto btn = CCMenuItemSpriteExtra::create(spr, this, menu_selector(RecPauseLayer::onRec));
        auto menu = CCMenu::create();
        menu->addChild(btn);
        menu->setID("rec-menu"_spr);
        menu->setPosition({win.width - 60.f, 30.f});
        this->addChild(menu, 100);
    }

    void onRec(CCObject*) {
        if (g_rec.on.load()) g_rec.stop();
        else g_rec.start();
        this->addRecButton();
    }
};
