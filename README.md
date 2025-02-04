# VehicularProject

You need python 3.8 or 3.9 version
You need to install Carla software and run it on your local machine

Steps to run the project:
1. python -m venv env1
2. .\env1\Scripts\activate
3. pip install -r requirements.txt
4. .\scripts\get_carla.sh
5. download yolov7.pt from here: https://github.com/WongKinYiu/yolov7
6. python train.py --workers 8 --device cpu --batch-size 32 --data data/carla.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights yolov7.pt --name yolov7 --htp data/hyp.scratch.p5.yaml
7. run carla.exe
8. Run classes to generate traffic and generate the car with the camera as the point of view
9. Run class generate_traffic.py first then run class carviewnoyolo.py
10. Record a Video or Image while the classes are running
11. python detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source testyolo.mp4