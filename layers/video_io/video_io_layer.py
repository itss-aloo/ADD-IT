
import cv2


def open_video(video_path):
    """
    Abre el vídeo de entrada.

    Devuelve:
    - objeto de captura
    - fps
    - resolución (width, height)
    """
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {video_path}")

    fps = float(video_capture.get(cv2.CAP_PROP_FPS))
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return video_capture, fps, (width, height)

def read_next_frame(video_capture):
    """
    Lee el siguiente frame del vídeo.

    Devuelve:
    - frame (imagen)
    - flag de fin de vídeo
    """
    success, frame = video_capture.read()
    end_of_video = not success or frame is None
    return frame, end_of_video

def initialize_video_writer(output_path, fps, frame_size):
    """
    Inicializa el escritor de vídeo de salida.

    Define:
    - ruta de salida
    - fps
    - tamaño de frame
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_path, fourcc, float(fps), tuple(frame_size))
    if not video_writer.isOpened():
        raise RuntimeError(f"No se pudo inicializar el writer de video: {output_path}")
    return video_writer

def write_frame(video_writer, frame):
    """
    Escribe un frame procesado en el vídeo de salida.
    """
    if video_writer is None:
        raise ValueError("video_writer no puede ser None")
    if frame is None or frame.size == 0:
        raise ValueError("frame no puede estar vacío")

    video_writer.write(frame)

def release_video_resources(video_capture, video_writer):
    """
    Libera recursos de vídeo (entrada y salida).
    """
    if video_capture is not None:
        video_capture.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()

def display_frame(frame):
    """
    Muestra el frame en una ventana (debug).
    """
    if frame is None or frame.size == 0:
        return

    cv2.imshow("ADD-IT Debug", frame)
    cv2.waitKey(1)
