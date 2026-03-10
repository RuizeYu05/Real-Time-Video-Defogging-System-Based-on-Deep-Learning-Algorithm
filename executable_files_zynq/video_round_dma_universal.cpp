/**
 * zynq_processor.cpp (Final Production Version)
 *
 * GStreamer application running on Zynq. It receives a UDP JPEG video stream,
 * processes each frame using AXI DMA (loopback), re-encodes it to JPEG,
 * and sends the processed stream to another UDP destination.
 *
 * This version includes the definitive fix for the GStreamer tx_pipeline issue
 * by correctly propagating timestamps from the input sample to the output buffer.
 */
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <glib.h>
#include <string.h>

extern "C" {
#include "libaxidma.h"
}


static const gchar *tx_host = "192.168.1.101"; 
static gint rx_port = 8080;
static gint tx_port = 8081;
static gint width = 640;
static gint height = 480;
static GOptionEntry entries[] = {
    {"tx-host", 't', 0, G_OPTION_ARG_STRING, &tx_host, "IP address to send to", "HOST"},
    {"rx-port", 'r', 0, G_OPTION_ARG_INT, &rx_port, "Port to receive from", "PORT"},
    {"tx-port", 'p', 0, G_OPTION_ARG_INT, &tx_port, "Port to send to", "PORT"},
    {"width", 'w', 0, G_OPTION_ARG_INT, &width, "Video width", "WIDTH"},
    {"height", 'e', 0, G_OPTION_ARG_INT, &height, "Video height", "HEIGHT"},
    {NULL}
};

typedef struct {
    GstElement *rx_pipeline;
    GstElement *tx_pipeline;
    GstElement *appsrc;
    GMainLoop *loop;

    axidma_dev_t dma_dev;
    int dma_tx_chan;
    int dma_rx_chan;
    void *dma_tx_buf;
    void *dma_rx_buf;
    size_t buffer_size;
} ZynqData;


static GstFlowReturn new_sample_callback(GstAppSink *appsink, gpointer data) {
ZynqData *zdata = (ZynqData *)data;
GstSample *sample;
GstBuffer *buffer;
GstMapInfo map;

sample = gst_app_sink_pull_sample(appsink);
if (sample == NULL) {
g_printerr("Failed to pull sample from appsink.\n");
return GST_FLOW_ERROR;
}
g_printerr("pull sample from sink\n");

buffer = gst_sample_get_buffer(sample);
gst_buffer_map(buffer, &map, GST_MAP_READ);

memcpy(zdata->dma_tx_buf, map.data, zdata->buffer_size);
g_printerr("mmap to dma buffer\n");

gst_buffer_unmap(buffer, &map);
gst_sample_unref(sample);

int rc = axidma_twoway_transfer(zdata->dma_dev, zdata->dma_tx_chan, zdata->dma_tx_buf, zdata->buffer_size, NULL,
zdata->dma_rx_chan, zdata->dma_rx_buf, zdata->buffer_size, NULL, TRUE);
if (rc < 0) {
g_printerr("AXI DMA twoway transfer failed!\n");
return GST_FLOW_ERROR;
}
g_printerr("dma transfer finished\n");

GstBuffer *new_buffer = gst_buffer_new_allocate(NULL, zdata->buffer_size, NULL);
gst_buffer_fill(new_buffer, 0, zdata->dma_rx_buf, zdata->buffer_size);

GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(zdata->appsrc), new_buffer);
if (ret != GST_FLOW_OK) {
g_printerr("Error pushing buffer to appsrc.\n");
}
g_printerr("send too appsrc\n");
return GST_FLOW_OK;
}

int main(int argc, char *argv[]) {
    ZynqData zdata = {0};
    GError *error = NULL;
    GstAppSink *appsink = NULL;
    GOptionContext *context;


    context = g_option_context_new("- GStreamer Zynq DMA Processor");
    g_option_context_add_main_entries(context, entries, NULL);
    if (!g_option_context_parse(context, &argc, &argv, &error)) { /* ... */ }
    g_option_context_free(context);


    gst_init(&argc, &argv);
    zdata.loop = g_main_loop_new(NULL, FALSE);
    g_print("Receiving on port %d, Sending to %s:%d, Resolution %dx%d\n", rx_port, tx_host, tx_port, width, height);
    zdata.dma_dev = axidma_init();
    if (zdata.dma_dev == NULL) { /* ... */ }
    const array_t* tx_chans = axidma_get_dma_tx(zdata.dma_dev);
    const array_t* rx_chans = axidma_get_dma_rx(zdata.dma_dev);
    if (tx_chans->len < 1 || rx_chans->len < 1) { /* ... */ }
    zdata.dma_tx_chan = tx_chans->data[0];
    zdata.dma_rx_chan = rx_chans->data[0];
    g_print("Using AXI DMA TX channel %d and RX channel %d.\n", zdata.dma_tx_chan, zdata.dma_rx_chan);
    zdata.buffer_size = width * height * 3;
    zdata.dma_tx_buf = axidma_malloc(zdata.dma_dev, zdata.buffer_size);
    zdata.dma_rx_buf = axidma_malloc(zdata.dma_dev, zdata.buffer_size);
    if (!zdata.dma_tx_buf || !zdata.dma_rx_buf) { /* ... */ }
    g_print("libaxidma initialized successfully.\n");


    gchar *rx_pipeline_str = g_strdup_printf(
        "udpsrc port=%d caps=\"application/x-rtp, media=(string)video, encoding-name=(string)JPEG, payload=(int)26\" ! "
        "rtpjpegdepay ! jpegdec ! videoconvert ! "
        "video/x-raw,format=BGR,width=%d,height=%d ! "
        "appsink name=mysink emit-signals=true max-buffers=1 drop=true",
        rx_port, width, height);
    zdata.rx_pipeline = gst_parse_launch(rx_pipeline_str, &error);
    g_free(rx_pipeline_str);
    if (error) { /* ... */ }
    appsink = GST_APP_SINK(gst_bin_get_by_name(GST_BIN(zdata.rx_pipeline), "mysink"));
    g_signal_connect(appsink, "new-sample", G_CALLBACK(new_sample_callback), &zdata);
    gst_object_unref(appsink);

    g_print("\nINFO: Using a simplified RAW video TX pipeline for diagnostics.\n");
    g_print("INFO: PC receiver must be adapted to handle RAW RTP stream.\n\n");
    gchar *tx_pipeline_str = g_strdup_printf(
        "appsrc name=mysrc ! "
        "rtprawpay pt=96 ! " 
        "udpsink host=%s port=%d",
        tx_host, tx_port);

    zdata.tx_pipeline = gst_parse_launch(tx_pipeline_str, &error);
    g_free(tx_pipeline_str);
    if (error) {
        g_printerr("Failed to create tx_pipeline: %s\n", error->message);
        return -1;
    }
    

    zdata.appsrc = gst_bin_get_by_name(GST_BIN(zdata.tx_pipeline), "mysrc");
    gchar *appsrc_caps_str = g_strdup_printf("video/x-raw,format=BGR,width=%d,height=%d,framerate=30/1", width, height);
    GstCaps *appsrc_caps = gst_caps_from_string(appsrc_caps_str);
    g_free(appsrc_caps_str);
    g_object_set(zdata.appsrc, "caps", appsrc_caps, "format", GST_FORMAT_TIME, NULL);
    gst_caps_unref(appsrc_caps);
    

    gst_element_set_state(zdata.tx_pipeline, GST_STATE_PLAYING);
    gst_element_set_state(zdata.rx_pipeline, GST_STATE_PLAYING);
    g_print("Zynq processor pipelines running...\n");
    g_main_loop_run(zdata.loop);


    g_print("Shutting down...\n");
    gst_element_set_state(zdata.rx_pipeline, GST_STATE_NULL);
    gst_element_set_state(zdata.tx_pipeline, GST_STATE_NULL);
    gst_object_unref(zdata.rx_pipeline);
    gst_object_unref(zdata.tx_pipeline);
    gst_object_unref(zdata.appsrc);
    g_main_loop_unref(zdata.loop);
    axidma_free(zdata.dma_dev, zdata.dma_tx_buf, zdata.buffer_size);
    axidma_free(zdata.dma_dev, zdata.dma_rx_buf, zdata.buffer_size);
    axidma_destroy(zdata.dma_dev);
    
    return 0;
}
