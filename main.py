import cv2
import matplotlib.pyplot as plt
import dlib


def handle_close(event, cap):
    cap.release()


def remove_noise(image):
    mask = cv2.getGaussianKernel(5, 0)
#    return cv2.filter2D(image, -1, mask)
    return cv2.GaussianBlur(image, (9, 9), 0)
#    return cv2.bilateralFilter(image, 5, 50, 0)


def detectFace(image):
    face_cascade = cv2.CascadeClassifier()
    #Load cascades
    if not face_cascade.load(cv2.samples.findFile('haarcascade_frontalface_default.xml')):
        print('Error - file not found\n')
        exit(0)

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    frame_gray = cv2.equalizeHist(gray_image)

    (a, b) = frame_gray.shape
    minsize = 200

    faces = face_cascade.detectMultiScale(remove_noise(frame_gray))
    #faces = face_cascade.detectMultiScale(remove_noise(frame_gray), 1, 0, 0, minsize, b) #-> non funziona :(

    for (x, y, w, h) in faces:
        image = cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0))

    return image


def grab_frame(cap):
    ret, frame = cap.read()
    return frame


def main():
    cap = cv2.VideoCapture(0)  # 0 o nessun parametro = webcam principale

    plt.ion()  # attiva modalità interattiva (serve per video perchè di base lavora con img)

    fig = plt.figure()  # serve crearla e mantenerla fissa perché altrimenti plt la chiude e riapre a ogni frame
    fig.canvas.mpl_connect("close_event", lambda event: handle_close(event, cap))

    # prepare a variable for the first run
    ax_img = None

    while cap.isOpened():
        frame = grab_frame(cap)  # cattura frame attuale
        #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if ax_img is None:
            frame = detectFace(frame)
            ax_img = plt.imshow(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # apply canny to first frame
            plt.axis('off')
            plt.title('Camera capture')
            plt.show()  # DO NOT forget this
        else:
            frame = detectFace(frame)
            ax_img.set_data(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # set current frame as data to show
            # update the figure
            fig.canvas.draw()
            fig.canvas.flush_events()  # serve perché interrompendo prog durante flusso video potrebbero esserci eventi in coda da cancellare
            plt.pause(1 / 24)  # 24 frame per second


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit(1)
