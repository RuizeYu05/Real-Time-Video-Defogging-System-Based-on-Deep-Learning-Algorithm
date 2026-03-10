#include <gst/gst.h>
#include <glib.h>

// 默认参数配置
static gchar *host = "192.168.1.102"; 
static gint port = 8080;
static gint width = 640;
static gint height = 480;
static gchar *camera_name = NULL;

// 命令行参数定义
static GOptionEntry entries[] = {
    {"host", 'h', 0, G_OPTION_ARG_STRING, &host, "IP address to send to", "HOST"},
    {"port", 'p', 0, G_OPTION_ARG_INT, &port, "Port to send to", "PORT"},
    {"width", 'w', 0, G_OPTION_ARG_INT, &width, "Video width", "WIDTH"},
    {"height", 'e', 0, G_OPTION_ARG_INT, &height, "Video height", "HEIGHT"},
    {"camera", 'c', 0, G_OPTION_ARG_STRING, &camera_name, "Camera display name (use 'list' to see options)", "NAME"},
    {NULL}
};

typedef struct {
    GstElement *pipeline;
    GMainLoop *loop;
} AppData;

// 消息总线回调
static gboolean bus_call(GstBus *bus, GstMessage *msg, gpointer data) {
    AppData *app = (AppData *)data;
    switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_EOS:
            g_print("End of stream\n");
            g_main_loop_quit(app->loop);
            break;
        case GST_MESSAGE_ERROR: {
            gchar *debug;
            GError *error;
            gst_message_parse_error(msg, &error, &debug);
            g_free(debug);
            g_printerr("Error: %s\n", error->message);
            g_error_free(error);
            g_main_loop_quit(app->loop);
            break;
        }
        default:
            break;
    }
    return TRUE;
}

// 列出设备函数
static void list_devices() {
    g_print("Discovering cameras...\n");
    GstDeviceMonitor *monitor = gst_device_monitor_new();
    gst_device_monitor_add_filter(monitor, "Video/Source", NULL);
    gst_device_monitor_start(monitor);
    GList *devices = gst_device_monitor_get_devices(monitor);
    if (!devices) {
        g_print("No cameras found.\n");
    } else {
        g_print("Available cameras:\n");
        for (GList *l = devices; l != NULL; l = l->next) {
            GstDevice *device = (GstDevice *)l->data;
            gchar *name = gst_device_get_display_name(device);
            g_print("  - \"%s\"\n", name);
            g_free(name);
        }
        g_list_free_full(devices, g_object_unref);
    }
    gst_device_monitor_stop(monitor);
    g_object_unref(monitor);
}

int main(int argc, char *argv[]) {
    AppData app;
    GstBus *bus;
    guint bus_watch_id;
    GError *error = NULL;
    GOptionContext *context;
    
    // 声明所有需要的 Element
    GstElement *pipeline, *source, *capsfilter, *tee;
    // 网络分支 Element
    GstElement *net_queue, *net_convert, *encoder, *payloader, *sink;
    // 显示分支 Element
    GstElement *disp_queue, *disp_convert, *textoverlay, *autovideosink;

    context = g_option_context_new("- GStreamer Camera Sender & Viewer");
    g_option_context_add_main_entries(context, entries, NULL);
    if (!g_option_context_parse(context, &argc, &argv, &error)) {
        g_printerr("Error parsing options: %s\n", error->message);
        g_option_context_free(context);
        return -1;
    }
    g_option_context_free(context);
    
    gst_init(&argc, &argv);

    if (camera_name && g_strcmp0(camera_name, "list") == 0) {
        list_devices();
        return 0;
    }

    // 1. 创建 Pipeline 和通用源
    pipeline = gst_pipeline_new("camera-pipeline");
    source = gst_element_factory_make("ksvideosrc", "source");
    capsfilter = gst_element_factory_make("capsfilter", "capsfilter");
    tee = gst_element_factory_make("tee", "stream-splitter"); // 关键：分流器

    // 2. 创建网络发送分支 Elements
    net_queue = gst_element_factory_make("queue", "net-queue"); // 关键：独立线程
    net_convert = gst_element_factory_make("videoconvert", "net-converter");
    encoder = gst_element_factory_make("jpegenc", "encoder");
    payloader = gst_element_factory_make("rtpjpegpay", "payloader");
    sink = gst_element_factory_make("udpsink", "sink");

    // 3. 创建本地显示分支 Elements
    disp_queue = gst_element_factory_make("queue", "disp-queue"); // 关键：独立线程
    disp_convert = gst_element_factory_make("videoconvert", "disp-converter");
    textoverlay = gst_element_factory_make("textoverlay", "overlay"); // 关键：文字叠加
    autovideosink = gst_element_factory_make("autovideosink", "display-sink"); // 关键：自动选择显示后端

    // 检查所有组件是否创建成功
    if (!pipeline || !source || !capsfilter || !tee || 
        !net_queue || !net_convert || !encoder || !payloader || !sink ||
        !disp_queue || !disp_convert || !textoverlay || !autovideosink) {
        g_printerr("Not all elements could be created.\n");
        return -1;
    }
    app.pipeline = pipeline;

    // --- 配置组件 ---

    // 配置摄像头
    if (camera_name) {
        g_object_set(G_OBJECT(source), "device-name", camera_name, NULL);
        g_print("Using camera: \"%s\"\n", camera_name);
    }

    // 配置网络目标
    g_object_set(G_OBJECT(sink), "host", host, "port", port, NULL);

    // 配置文字叠加 (显示 "Raw Capture")
    // valignment=2 (top), halignment=0 (left), font-desc 设置字体大小
    g_object_set(G_OBJECT(textoverlay), 
        "text", "Local Raw Capture (Direct)", 
        "valignment", 2, // 2 = GST_TEXT_OVERLAY_VALIGN_TOP
        "halignment", 0, // 0 = GST_TEXT_OVERLAY_HALIGN_LEFT
        "font-desc", "Sans, 20",
        "shaded-background", TRUE, // 添加半透明背景以便文字清晰
        NULL);
    
    // 配置分辨率
    gchar *caps_str = g_strdup_printf("video/x-raw,width=%d,height=%d,framerate=30/1", width, height);
    GstCaps *caps = gst_caps_from_string(caps_str);
    g_free(caps_str);
    g_object_set(G_OBJECT(capsfilter), "caps", caps, NULL);
    gst_caps_unref(caps);

    // --- 添加组件到 Pipeline ---
    gst_bin_add_many(GST_BIN(pipeline), 
        source, capsfilter, tee,                  // 公共部分
        net_queue, net_convert, encoder, payloader, sink, // 网络分支
        disp_queue, disp_convert, textoverlay, autovideosink, // 显示分支
        NULL);

    // --- 连接组件 ---
    
    // 1. 连接源到 Tee (source -> caps -> tee)
    if (!gst_element_link_many(source, capsfilter, tee, NULL)) {
        g_printerr("Source elements could not be linked.\n");
        gst_object_unref(pipeline);
        return -1;
    }

    // 2. 连接网络分支 (tee -> queue -> convert -> jpeg -> rtp -> udp)
    // 注意：GStreamer 足够智能，link_many 会自动处理 tee 的 request pad
    if (!gst_element_link_many(tee, net_queue, net_convert, encoder, payloader, sink, NULL)) {
        g_printerr("Network branch could not be linked.\n");
        gst_object_unref(pipeline);
        return -1;
    }

    // 3. 连接显示分支 (tee -> queue -> convert -> overlay -> display)
    if (!gst_element_link_many(tee, disp_queue, disp_convert, textoverlay, autovideosink, NULL)) {
        g_printerr("Display branch could not be linked.\n");
        gst_object_unref(pipeline);
        return -1;
    }

    // --- 运行循环 ---
    bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
    bus_watch_id = gst_bus_add_watch(bus, bus_call, &app);
    gst_object_unref(bus);
    app.loop = g_main_loop_new(NULL, FALSE);

    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    g_print("Pipeline running:\n");
    g_print(" -> Sending JPEG to %s:%d\n", host, port);
    g_print(" -> Displaying local window with overlay.\n");
    
    g_main_loop_run(app.loop);

    // --- 清理 ---
    g_print("Shutting down.\n");
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(GST_OBJECT(pipeline));
    g_source_remove(bus_watch_id);
    g_main_loop_unref(app.loop);

    return 0;
}