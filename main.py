from imageai.Detection import VideoObjectDetection
import os

# Получаем путь рабочей директории
execution_path = os.getcwd()
# Путь к файлу с моделью сети
model_path = execution_path + "/yolov3_.pt"
# Путь к файлу с видео
vide_path_in = execution_path + "/file_full.mp4"
# Место куда выводиться результат обработки
vide_path_out = execution_path + "/traffic_detected"

detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(model_path)
detector.loadModel()

# Запускаем детекцию объектов
video_path = detector.detectObjectsFromVideo(
    input_file_path=vide_path_in,
    output_file_path=vide_path_out,
    frames_per_second=20,
    log_progress=True)
# Выводим путь к видео
print(video_path)
