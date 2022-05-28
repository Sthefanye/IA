from tkinter import *
import cv2
from tkinter import messagebox
import os
import numpy
import engine
# Grupo: 
#   Gabriela Prestes Farias | Matrícula: 03058613
#   George Moraes de Sousa | Matrícula: 03113538
#   José Ribamar Queiroz da Silva Neto | Matrícula: 03118421
#   Sthefanye Guimarães Oliveira | Matrícula: 03116527

class ReconhecimentoFacial():
    bg_color = '#2b2c49'

    #Menu de opções.
    def __init__(self, root):
        self.root = root
        self.haar_file = 'haarcascade_frontalface_alt2.xml'

        #Título do menu.
        title = Label(self.root, bg=self.bg_color, text=' Reconhecimento Facial', fg="#ffffff",
                      font=('arial', 15, 'bold'), height=3)
        title.pack(fill=BOTH)

        #Container principal.
        control_frame = Frame(self.root, height=150, bg=self.bg_color, bd=2, relief='ridge')
        control_frame.pack(pady=20, fill=BOTH, padx=10)

        #Botão de cadastrar a face do usuário.
        train_button = Button(control_frame, text='Cadastro', width=12,
                              bd=2, height=3, relief=GROOVE, font=('arial', 12, 'bold'), bg="#4b96fe",fg="#ffffff", command=self.get_data)
        train_button.place(x=30, y=30)

        #Botão de realizar a leitura da face do usuário.
        test_button = Button(control_frame, text='Leitura', bd=2, width=12,
                             height=3, relief=GROOVE, font=('arial', 12, 'bold'), bg="#6f5afa", fg="#ffffff", command=self.ModeloTeste)
        test_button.place(x=175, y=30)

        #Botão de sair da IA.
        exit_button = Button(control_frame, text='Sair', width=12,
                             bd=2, height=3, relief=GROOVE, font=('arial', 12, 'bold'), bg="#ff7686", fg="#ffffff", command=root.quit)
        exit_button.place(x=320, y=30)

        # ------------------------------------function Defination ------------------------------------

    def modelo_treinamento(self):
        # Captura nome e ID que o usuário inserir.
        name_ = self.name.get()
        id_ = self.id_ent.get()
        print(name_, id_)
        self.top.destroy()
        self.take_images(name_, id_)


    #Tela de cadastro.
    def get_data(self):

        # Configuração do estilo e tamanho da tela.
        self.top = Toplevel()
        self.top.geometry('270x200+240+200')
        self.top.configure(bg='#454572')
        self.top.resizable(0, 0)

        # Input do nome.
        name_lbl = Label(self.top, text='Seu nome', width=10, font=('arial', 10, 'bold')).place(x=20, y=20)
        self.name = Entry(self.top, width=15, font=('arial', 12))
        self.name.place(x=115, y=20)
        
        # Input do ID.
        id_lbl = Label(self.top, text='Identificação', width=10, font=('arial', 10, 'bold')).place(x=20, y=60)
        self.id_ent = Entry(self.top, width=15, font=('arial', 12))
        self.id_ent.place(x=115, y=60)

        # Botão de iniciar o treinamento facial, onde captura a face do usuário.
        btn = Button(self.top, text='Treinamento Facial', font=('arial', 12, 'bold'),bg="#4b96fe",fg="#ffffff", command=self.modelo_treinamento)
        btn.place(x=55, y=120)

    # Método que realiza o reconhecimento facial.
    def ModeloTeste(self):
        datasets = 'dataset'

        # Cria uma lista de imagens
        (images, lables, names, id) = ([], [], {}, 0)

        # Carregar a lista de imagens do datasets -> o datasets armazena as imagens de treinamento.
        for (subdirs, dirs, files) in os.walk(datasets):
            for subdir in dirs:
                names[id] = subdir
                subjectpath = os.path.join(datasets, subdir)
                for filename in os.listdir(subjectpath):
                    path = subjectpath + '/' + filename
                    lable = id
                    images.append(cv2.imread(path, 0))
                    lables.append(int(lable))
                id += 1

        # O tamanho das imagens de treinamento, qualquer imagem inserida, precisa ter este tamanho.
        (width, height) = (130, 100)

        # Cria um array Numpy para as duas lista acima.
        (images, lables) = [numpy.array(lis) for lis in [images, lables]]

        # OpenCV treinamento das imagens
        model = cv2.face.LBPHFaceRecognizer_create()
        model.train(images, lables)

        # Usar o xml para fazer o reconhecimento de qualquer objeto na tela.
        face_cascade = cv2.CascadeClassifier(self.haar_file)

         # Abrir a camera e captutar as imagens
        webcam = cv2.VideoCapture(0)

        # Treinamento da IA para reconhecimento facial com ou sem máscaras.

        dataframe = engine.load_dataframe() # Carregando dataframe com as imagens para treinamento.

        X_train, y_train = engine.train_test(dataframe) #Dividindo conjuntos de treino e teste.
        pca = engine.pca_model(X_train) # Modelo PCA para extração de features da imagem. 

        X_train = pca.transform(X_train) # Conjunto de treino com features extraídas.

        knn = engine.knn(X_train, y_train) # Treinando modelo classificatório KNN.

        # Rótulo das classificações
        label = {
            0: "Sem mascara",
            1: "Com mascara"
        }

        #Reconhecimento facial e detecção de mascara ou sem
        while True:
            #Leitura dos frames.
            (_, im) = webcam.read()
            
            #Converte a cores em escalas de cinza.
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            
            #Detecta as faces encontrada no frame.
            faces = face_cascade.detectMultiScale(gray)
            
            # Opção de sair ao clicar na tecla Q.
            if cv2.waitKey(1) & 0xff == ord('q'):
                break

            # Percorrendo as faces encontradas.
            for (x, y, w, h) in faces:
                face = gray[y:y + h, x:x + w]
                face_resize = cv2.resize(face, (width, height))
                # Tenta reconhecer a face
                prediction = model.predict(face_resize)

                classification = ""
                color = (0, 255, 0)
                
                # Lógivca com mascara ou não.
                if face.shape[0] >= 200 and face.shape[1] >= 200:
                    vector = pca.transform([face_resize.flatten()]) #Extraindo features da imagem.
                    pred = knn.predict(vector)[0] # Tenta identificar se está com máscara ou não.
                    classification = label[pred] # Busca a label conforme a identificação.

                    # Alterando a cor do retangulo caso esteja sem mascara.
                    if pred == 0:
                        color = (0,0,255)

                #Lógica do reconhecimento facial
                # Retângulo ao redor do rosto do usuário.
                cv2.rectangle(im, (x, y), (x + w, y + h), color, 3)

                if prediction[1] < 120: # O valor que calibra o reconhecimento, quanto menor o valor, mais preciso é a leitura da imagem gravada no treinamento.
                    cv2.putText(im, '% s - %.0f - % s' % # Mostrando o nome, ID, e classificação se está com máscara ou sem do usuário.
                                (names[prediction[0]], prediction[1], classification), (x - 10, y - 10),
                                cv2.FONT_HERSHEY_PLAIN, 1, color)
                else:
                    # Se a face não for reconhecida, mostra o desconhecido, e ainda detecta se está com máscara ou sem.
                    cv2.putText(im, 'Desconhecido - % s' % (classification),
                                (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, color)

            # Exibir uma imagem em uma janela.
            cv2.imshow('Leitura Facial', im)

        # Permite que os usuário destrua todas as janelas a qualquer momento.
        cv2.destroyAllWindows()


    def take_images(self,name_,id_):
        # time.sleep(2)
        # Todas as imagens ficam na pasta dataset.
        datasets = 'dataset'
        # Criar subpastas com o nome.
        sub_data = str(name_)+ '-' + str(id_)
        path = os.path.join(datasets, sub_data)
        if not os.path.isdir(path):
            os.mkdir(path)

        # Definir tamanho das imagens
        (width, height) = (130, 100)

        # '0' é usado para webcam
        face_cascade = cv2.CascadeClassifier(self.haar_file)
        webcam = cv2.VideoCapture(0)

        # CADASTRO DA FACE
        # Tente tirar até 30 fotos do usuário.
        count = 1
        while count < 30:
            (_, im) = webcam.read() # Leitura da imagem na webcam
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) # Converte as imagens para escala de cinza
            faces = face_cascade.detectMultiScale(gray, 1.3, 4)  # Detecta multiplas faces
            for (x, y, w, h) in faces:
                cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
                face = gray[y:y + h, x:x + w] # Face capturada
                face_resize = cv2.resize(face, (width, height)) # Dimensões das imagens capturadas.
                cv2.imwrite('% s/% s.png' % (path, count), face_resize) # Transforma a face capturada em imagem do tipo png.
            count += 1

            cv2.imshow('Cadastrando face', im)
            key = cv2.waitKey(10)
            if key == 27:
                break
        cv2.destroyAllWindows()
        messagebox.showinfo("Face Cadastrada","Modelo foi treinado a imagem \n  Você será reconhecido.")

# Configuração da janela do programa
if __name__ == '__main__':
    root = Tk()
    ReconhecimentoFacial(root)
    root.geometry('500x250+240+200')
    root.title("ARE 2 - IA")
    root.resizable(0, 0)
    root.configure(bg=ReconhecimentoFacial.bg_color)
    root.mainloop()
