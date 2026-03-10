#include <gst/gst.h>
#include <glib.h>


static gint port = 8081;


static GOptionEntry entries[] = {
    {"port", 'p', 0, G_OPTION_ARG_INT, &port, "Port to receive from", "PORT"},
    {NULL}
};

typedef struct {
    GstElement *pipeline;
    GMainLoop *loop;
} AppData;

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

int main(int argc, char *argv[]) {
    AppData app;
    GstBus *bus;
    guint bus_watch_id;
    GError *error = NULL;
    GOptionContext *context;
    GstElement *pipeline, *source, *depayloader, *decoder, *converter, *sink;

    context = g_option_context_new("- GStreamer Video Receiver for Windows (JPEG)");
    g_option_context_add_main_entries(context, entries, NULL);
    if (!g_option_context_parse(context, &argc, &argv, &error)) {
        g_printerr("Error parsing options: %s\n", error->message);
        g_option_context_free(context);
        return -1;
    }
    g_option_context_free(context);

    gst_init(&argc, &argv);

    pipeline = gst_pipeline_new("receiver-pipeline");
    source = gst_element_factory_make("udpsrc", "source");
    depayloader = gst_element_factory_make("rtpjpegdepay", "depayloader");
    decoder = gst_element_factory_make("jpegdec", "decoder");
    converter = gst_element_factory_make("videoconvert", "converter");
    sink = gst_element_factory_make("autovideosink", "sink");

    if (!pipeline || !source || !depayloader || !decoder || !converter || !sink) {
        g_printerr("Not all elements could be created. Check GStreamer installation.\n");
        return -1;
    }
    app.pipeline = pipeline;

    g_object_set(G_OBJECT(source), "port", port, NULL);

    GstCaps *rtp_caps = gst_caps_from_string(
        "application/x-rtp, media=(string)video, encoding-name=(string)JPEG, payload=(int)26");
    g_object_set(G_OBJECT(source), "caps", rtp_caps, NULL);
    gst_caps_unref(rtp_caps);

    gst_bin_add_many(GST_BIN(pipeline), source, depayloader, decoder, converter, sink, NULL);

    if (!gst_element_link_many(source, depayloader, decoder, converter, sink, NULL)) {
        g_printerr("Elements could not be linked.\n");
        gst_object_unref(pipeline);
        return -1;
    }

    bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
    bus_watch_id = gst_bus_add_watch(bus, bus_call, &app);
    gst_object_unref(bus);
    app.loop = g_main_loop_new(NULL, FALSE);

    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    g_print("Receiving JPEG on port %d...\n", port);
    g_main_loop_run(app.loop);

    g_print("Shutting down.\n");
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(GST_OBJECT(pipeline));
    g_source_remove(bus_watch_id);
    g_main_loop_unref(app.loop);

    return 0;
}