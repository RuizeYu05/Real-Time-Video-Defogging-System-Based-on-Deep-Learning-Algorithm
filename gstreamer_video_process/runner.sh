./receiver.exe --port 8081
./sender.exe --host 192.168.1.102 --port 8080 --width 640 --height 480 --camera "HD Pro Webcam C920"
./sender.exe --camera list
./gstreamer-processor_universal --rx-port 8080 --tx-host 192.168.1.100 --tx-port 8081 --width 640 --height 480

