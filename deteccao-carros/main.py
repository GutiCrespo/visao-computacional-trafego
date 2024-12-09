import cv2
import numpy as np
import time
from datetime import datetime
# Variável para definir se o modelo é YOLOv3-tiny ou não
TINY = False

# Variáveis para Alterar 

# Média para verificação
RUNNING_MEAN_SECONDS = 10 

# Configurações do modelo YOLOv3 ou YOLOv3-tiny
ARQUIVO_CFG = "deteccao-carros/yolov3{}.cfg".format("-tiny" if TINY else "")
ARQUIVO_PESOS = "deteccao-carros/yolov3{}.weights".format("-tiny" if TINY else "")
ARQUIVO_CLASSES = "deteccao-carros/coco{}.names".format("-tiny" if TINY else "")

# Carregar os nomes das classes
with open(ARQUIVO_CLASSES, "r") as arquivo:
    CLASSES = [linha.strip() for linha in arquivo.readlines()]

# Gerar cores diferentes para cada classe
CORES = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Definir a entrada de vídeo a ser usada pelo modelo
VIDEO_FOOTAGE = "deteccao-carros/video-footage.mp4"

# Cor das informações na Tela
WHITE = [255, 255, 255] 
BLACK = [0, 0, 0]       
PURPLE = [128, 0, 128]  
RED = [0, 0, 255]       
ALL_COLORS = BLACK

def carregar_modelo_pretreinado():
    """
    Carrega o modelo YOLOv3 pré-treinado e configurações associadas ao OpenCV.
    """
    modelo = cv2.dnn.readNetFromDarknet(ARQUIVO_CFG, ARQUIVO_PESOS)
    modelo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    modelo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    if modelo.empty():
        raise IOError("Não foi possível carregar o modelo de detecção de objetos.")
        
    return modelo

def preprocessar_frame(frame):
    """
    Pré-processa o frame para detecção: redimensiona e normaliza.
    """
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
    return blob

def detectar_objetos(frame, modelo):
    """
    Detecta objetos no frame usando o modelo carregado.
    """
    blob = preprocessar_frame(frame)
    modelo.setInput(blob)
    nomes_camadas = modelo.getLayerNames()
    camadas_saida = [nomes_camadas[i - 1] for i in modelo.getUnconnectedOutLayers()]
    saidas = modelo.forward(camadas_saida)
    return saidas

def desenhar_deteccoes(
    frame, 
    deteccoes, 
    limiar: int,
    avg: float,
    fps: int,
    tendency: str,
):
    """
    Desenha retângulos ao redor dos objetos detectados com confiança acima do limiar.
    """
    (altura, largura) = frame.shape[:2]
    caixas = []
    confiancas = []
    ids_classes = []
    for saida in deteccoes:
        for deteccao in saida:
            pontuacoes = deteccao[5:]
            id_classe = np.argmax(pontuacoes)
            confianca = pontuacoes[id_classe]

            # Se for carro e maior do que o nível míninimo definido no limiar
            if id_classe == 2 and confianca > limiar:
                caixa = deteccao[0:4] * np.array([largura, altura, largura, altura])
                (centroX, centroY, largura_caixa, altura_caixa) = caixa.astype("int")
                x = int(centroX - (largura_caixa / 2))
                y = int(centroY - (altura_caixa / 2))
                caixas.append([x, y, int(largura_caixa), int(altura_caixa)])
                confiancas.append(float(confianca))
                ids_classes.append(id_classe)

    indices = cv2.dnn.NMSBoxes(caixas, confiancas, limiar, limiar - 0.1)
    
    num_vehicles = 0

    if len(indices) > 0:
        num_vehicles = len(indices)
        for i in indices.flatten():
            klass = ids_classes[i]
            (x, y) = (caixas[i][0], caixas[i][1])
            (largura_caixa, altura_caixa) = (caixas[i][2], caixas[i][3])
            cor = [int(c) for c in CORES[klass]]
            cv2.rectangle(frame, (x, y), (x + largura_caixa, y + altura_caixa), cor, 2)
            texto = f"{CLASSES[klass]}: {confiancas[i]:.2f}"
            cv2.putText(frame, texto, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)
    
    # Infos na janela:
    TIME = obter_horario_formatado()
    cv2.putText(frame, f"FPS: {fps:.2f}", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ALL_COLORS, 2) # FPS
    cv2.putText(frame, f"{TIME}", (50, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ALL_COLORS, 2) # Tendência da Rodovia
    
    cv2.putText(frame, f"Num vehicles in pic: {num_vehicles}", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ALL_COLORS, 2) # Número de Veículos
    cv2.putText(frame, f"Avg in last {RUNNING_MEAN_SECONDS} seconds: {round(avg)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ALL_COLORS, 2) # Média nos últimos "X" segundos
    cv2.putText(frame, f"TENDENCY: {tendency}", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ALL_COLORS, 2) # Tendência da Rodovia
            
    return num_vehicles

from datetime import datetime

def obter_horario_formatado():
    """
    Retorna o horário atual
    """
    now = datetime.now()
    return now.strftime("%d/%m/%Y - %H:%M:%S")


def main():
    """
    Executa a detecção de objetos em tempo real usando a webcam.
    """
    print("Buscando fonte de vídeo...")
    modelo = carregar_modelo_pretreinado()
    captura_video = cv2.VideoCapture(VIDEO_FOOTAGE)

    if not captura_video.isOpened():
        raise Exception("Não foi possível encontrar a entrada de vídeo.")

    # Reduzir a resolução do vídeo capturado
    captura_video.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    captura_video.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    limiar_confianca = 0.5 # valor inicial do limiar de confiança


    def ajustar_limiar(valor):
        nonlocal limiar_confianca
        limiar_confianca = valor / 100

    cv2.namedWindow('Detecta Objetos')
    cv2.createTrackbar('Limiar de Confiança', 'Detecta Objetos', int(limiar_confianca * 100), 100, ajustar_limiar)


    vehicle_counts = []
    fps = 0
    tendency = "STABLE"  # Tendência inicial
    PILE_SIZE = 3  # Limite de quantas médias queremos usar para criar a tendência

    try:
        last_vehicle_count = 0
        long_term_avg = 0  # Inicializa a média de longo prazo
        smoothing_factor = 0.1  # Fator de suavização para ajustar long_term_avg e não mudar do nada

        while True:
            start = time.time()
            ret, frame = captura_video.read()
            if not ret:
                break

            # Atualiza a lista de contagens recentes
            vehicle_counts = [(count, tm) for (count, tm) in vehicle_counts if tm > (start - RUNNING_MEAN_SECONDS)]
            cts = [c for c, t in vehicle_counts]
            running_avg = sum(cts) / len(cts) if len(cts) > 0 else 0

            # Atualiza a média de longo prazo (com a suavização já prevista)
            if long_term_avg == 0:
                long_term_avg = running_avg 
            else:
                long_term_avg = (smoothing_factor * running_avg) + ((1 - smoothing_factor) * long_term_avg)

            # Detectar a tendência
            relative_threshold = max(long_term_avg * 0.1, 2)  # Limiar dinâmico (mínimo absoluto de 2)

            # Verificação explícita para fluxos muito baixos
            if running_avg <= 1 and long_term_avg > 5:  # Detecção de fluxo muito baixo
                tendency = "DOWN"
            elif running_avg > long_term_avg + relative_threshold:  # Aumento significativo
                tendency = "UP"
            elif running_avg < long_term_avg - relative_threshold:  # Diminuição significativa
                tendency = "DOWN"
            else:
                tendency = "STABLE"

            # Detecção de veículos no frame atual
            deteccoes = detectar_objetos(frame, modelo)
            vehicle_count = desenhar_deteccoes(frame, deteccoes, limiar_confianca, running_avg, fps, tendency)

            # Atualiza o histórico de contagens
            vehicle_counts.insert(0, (vehicle_count, start))

            # Exibe o frame com informações
            cv2.imshow('Detecta Objetos', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Atualiza FPS
            end = time.time()
            delta = end - start
            fps = 1 / delta
    finally:
        captura_video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
