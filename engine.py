import face_recognition as fr

def reconhece_face(url_foto):
    foto = fr.load_image_file(url_foto)
    rostos = fr.face_encodings(foto)
    if(len(rostos) > 0):
        return True, rostos
    
    return False, []

def get_rostos():
    rostos_conhecidos = []
    nomes_dos_rostos = []

    sthefanye1 = reconhece_face("./img/sthefanye1.jpg")
    if(sthefanye1[0]):
        rostos_conhecidos.append(sthefanye1[1][0])
        nomes_dos_rostos.append("Sthefanye")

    neto1 = reconhece_face("./img/neto1.jpg")
    if(sthefanye1[0]):
        rostos_conhecidos.append(neto1[1][0])
        nomes_dos_rostos.append("Franscisco Neto")


    return rostos_conhecidos, nomes_dos_rostos