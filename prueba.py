#Importar librerias
print("Cargando librerias...")
import cv2
import darknet
import argparse

#Cargar red
weights_red_path="redes/yolov7-tiny.weights"
cfg_red_path="./redes/yolov7-tiny.cfg"
red_thresh=.35

def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input", type=str, default="",
                        help="image source. It can be a single image, a"
                        "txt with paths to them, or a folder. Image valid"
                        " formats are jpg, jpeg or png."
                        "If no input is given, ")
    parser.add_argument("--batch_size", default=1, type=int,
                        help="number of images to be processed at the same time")
    parser.add_argument("--weights", default=weights_red_path,
                        help="yolo weights path")
    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--save_labels", action='store_true',
                        help="save detections bbox for each image in yolo format")
    parser.add_argument("--config_file", default=cfg_red_path,
                        help="path to config file")
    parser.add_argument("--data_file", default="./cfg/coco.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=red_thresh,
                        help="remove detections with lower confidence")
    return parser.parse_args()

args = parser()
print("Configuracion de red:")
print(args)

#Cargar red
print("Cargando red...")
network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=args.batch_size
    )

#Empezar video
print('Grabando Camara...')
cap = cv2.VideoCapture(0)

while cap:
    ret, image = cap.read()
    if ret is False:
        break

    #Preparando deteccion
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                                interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())

    #Detectar
    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=args.thresh)
    darknet.free_image(darknet_image)
    image = darknet.draw_boxes(detections, image_resized, class_colors)

    cv2.imshow('frame',cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

