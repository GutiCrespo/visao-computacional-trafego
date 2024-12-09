
# **Detecção de Fluxo de Veículos com YOLOv3 e OpenCV**

Projeto criado para o curso **Fundamentos de Inteligência Artificial**, pelos alunos Gustavo Crespo da Silva e Leontino de Melo Madruga. Ele utiliza **YOLOv3** e **OpenCV** para detectar veículos em vídeos em tempo real, exibindo informações sobre o fluxo de tráfego.

---

## **Descrição do Projeto**

O código realiza as seguintes funções:
1. Detecta veículos em tempo real a partir de um vídeo ou webcam.
2. Exibe:
   - A quantidade de veículos detectados em cada frame.
   - A média de veículos nos últimos **10 segundos**.
   - A **tendência do fluxo de tráfego**:
     - **UP:** Aumento no fluxo.
     - **DOWN:** Diminuição no fluxo.
     - **STABLE:** Fluxo constante.
   - A taxa de quadros por segundo (**FPS**).
   - O horário atual.

---

## **Requisitos do Projeto**

### **Arquivos Necessários**
Certifique-se de possuir os seguintes arquivos para o YOLOv3:
- Configuração do modelo (`yolov3.cfg`)
- Pesos treinados (`yolov3.weights`)
- Nomes das classes (`coco.names`)

Você pode baixá-los da [implementação oficial do YOLOv3](https://pjreddie.com/darknet/yolo/).

### **Dependências**
As bibliotecas utilizadas estão listadas no arquivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

### **Conteúdo do `requirements.txt`**
```text
numpy==2.0.0
opencv-python==4.10.0.84
```

---

## **Como Rodar o Projeto**

### 1. Configurar o Ambiente Virtual
Crie e ative um ambiente virtual:

- No macOS e Linux:
  ```bash
  python3 -m venv env
  source env/bin/activate
  ```

- No Windows:
  ```bash
  python -m venv env
  .\env\Scripts\activate
  ```

### 2. Instalar as Dependências
Certifique-se de que o ambiente virtual está ativo e instale as dependências:
```bash
pip install -r requirements.txt
```

### 3. Escolher o vídeo
Adicione o vídeo que você deseja analisar a tendência na variável 
```bash
VIDEO_FOOTAGE = "deteccao-carros/[SEU-VIDEO].mp4"
```
Você pode ainda apenas adicionar o vídeo à pasta `deteccao-carros` e alterar o nome para `video-footage.mp4`

### 4. Rodar o Projeto
Após configurar tudo, execute o arquivo principal:
```bash
python main.py
```

### Dica Extra:
Utilize o seletor de limiar para "aumentar e diminuir" o número de veículos na tela, para testar a funcionalidade de **tendência**

---

## **Funcionamento do Código**

O código está dividido em funções para maior organização e clareza. Abaixo, algumas das principais funcionalidades implementadas:

### **Detecção de Veículos**
- A função `detectar_objetos` utiliza o modelo YOLOv3 para identificar veículos em cada frame capturado.
- A função `desenhar_deteccoes` destaca os veículos detectados com caixas delimitadoras e exibe informações no frame.

### **Cálculo de Tendência**
- A tendência do fluxo de tráfego é calculada com base em:
  - **Média de curto prazo:** Quantidade média de veículos nos últimos **10 segundos**.
  - **Média de longo prazo:** Ajustada dinamicamente para refletir o fluxo geral.
- A lógica de tendência é exibida diretamente no frame:
  - **UP:** Quando há um aumento significativo no fluxo.
  - **DOWN:** Quando há uma redução significativa no fluxo.
  - **STABLE:** Quando o fluxo permanece constante.

### **Comentários no Código**
- Realizamos diversos comentários no próprio código, a fim de facilitar a compreensão de cada funcionalidade.
- Para detalhes técnicos do funcionamento do projeto, leia os comentários diretamente no código.

---
