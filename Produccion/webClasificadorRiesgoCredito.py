#------------------------------
#Autor: Mayron Rodriguez Sibaja. 113250371
#Fecha: Setiembre, 2023
#Descripcion: sitio web basico, para probar un modelo preentrenado de clasificacion binaria de riesgo de credito.
#Para la Universidad Complutense de Madrid por el trabajo de fin de Master big data & business analytics
#Modalidad: Online (2022-2023)
#------------------------------


import pandas as pd
import pickle
from sklearn import svm
from sklearn.pipeline import  Pipeline
import streamlit as st
import Preprocesar_ClasificadorRiesgoCredito
from Preprocesar_ClasificadorRiesgoCredito import TraducirVariables,RecodificarVariables,ImputacionVariables,TratarVariablesCatOrdinales, EscalarVariables,DummyVariables,SeleccionVariables

# Path del modelo preentrenado
MODEL_PATH = 'ClasificadorRiesgoCreditoAleman.pkl'

Clasificador=''
# Se carga el modelo
if Clasificador=='':
    with open(MODEL_PATH, 'rb') as file:
        Clasificador = pickle.load(file)
         

def model_prediction(x_in, Clasificador):
    preds=Clasificador.predict(x_in)
    return preds

def model_prediction_prob(x_in, Clasificador):
    predsProba=Clasificador.predict_proba(x_in)[:,1]
    return predsProba


def main():
    
    
   with st.container():
           # Título
    html_temp = """
    <h1 style="color:#181082;text-align:center;">SISTEMA DE CLASIFICACIÓN DE RIESGO DE CRÉDITO</h1>
    </div>
    <h2 style="color:#181082;text-align:center;">BANCO ALEMÁN</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.write("""Por favor ingrese los siguientes datos del solicitante de crédito""")  
    st.write("---")
    #Configurar opciones
    estado_opciones = {"sin cuenta registrada": 1, "… < 0 DM ": 2, "0<= … < 200 DM ": 3, "… >= 200 DM ":4}
    duracion_opciones = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10,
                        "11": 11, "12": 12, "13": 13, "14": 14, "15": 15, "16": 16, "17": 17, "18": 18, "19": 19, "20": 20,
                        "21": 21, "22": 22, "23": 23, "24": 24, "25": 25, "26": 26, "27": 27, "28": 28, "29": 29, "30": 30,
                        "31": 31, "32": 32, "33": 33, "34": 34, "35": 35, "36": 36, "37": 37, "38": 38, "39": 39, "40": 40,
                        "41": 41, "42": 42, "43": 43, "44": 44, "45": 45, "46": 46, "47": 47, "48": 48, "49": 49, "50": 50,
                        "51": 51, "52": 52, "53": 53, "54": 54, "55": 55, "56": 56, "57": 57, "58": 58, "59": 59, "60": 60,
                        "61": 61, "62": 62, "63": 63, "64": 64, "65": 65, "66": 66, "67": 67, "68": 68, "69": 69, "70": 70}
    historial_cred_opciones = { "retraso en el pago en el pasado":0,
                                "cuenta crítica/otros créditos en otro lugar":1,
                                "ningún crédito tomado/todos los créditos pagados puntualmente":2,
                                "créditos existentes pagados puntualmente hasta ahora":3,
                                "todos los créditos en este banco pagados puntualmente":4}
        
        
    proposito_opciones = {  "otros": 0,
                            "automóvil (nuevo)": 1,
                            "automóvil (usado)": 2,
                            "muebles/equipos": 3,
                            "radio/televisión": 4,
                            "electrodomésticos": 5,
                            "reparaciones": 6,
                            "educación": 7,
                            "vacaciones": 8,
                            "reentrenamiento": 9,
                            "negocio": 10}
        
      
    ahorros_opciones = {"desconocido/sin cuenta de ahorros": 1,
                        "... < 100 DM": 2,
                        "100 <= ... < 500 DM": 3,
                        "500 <= ... < 1000 DM": 4,
                        "... >= 1000 DM": 5}

    duracion_empleo_opciones = {"desempleado": 1,
                                "< 1 año": 2,
                                "1 <= ... < 4 años": 3,
                                "4 <= ... < 7 años": 4,
                                ">= 7 años": 5 }      
        
    tasa_pago_cuotas_opciones = {   ">= 35": 1,
                                    "25 <= ... < 35": 2,
                                    "20 <= ... < 25": 3,
                                    "< 20": 4 }
                                    
    estado_personal_sex_opciones = {"hombre: divorciado/separado": 1,
                                    "mujer: no soltera o hombre: soltero": 2,
                                    "hombre: casado/viudo": 3,
                                    "mujer: soltera": 4}
    
    otros_deudores_opciones = { "ninguno": 1,
                                "co-solicitante": 2,
                                "garante": 3}
    
    residencia_actual_opciones = {  "< 1 año": 1,
                                    "1 <= ... < 4 años": 2,
                                    "4 <= ... < 7 años": 3,
                                    ">= 7 años": 4}
                                    
    propiedad_opciones = {  "desconocido / sin propiedad": 1,
                            "automóvil u otro": 2,
                            "sociedad de construcción, ahorros agrarios / seguro de vida": 3,
                            "bienes raíces": 4}
    
    edad_opciones = {   "19 años": 19,
                        "20 años": 20,
                        "21 años": 21,
                        "22 años": 22,
                        "23 años": 23,
                        "24 años": 24,
                        "25 años": 25,
                        "26 años": 26,
                        "27 años": 27,
                        "28 años": 28,
                        "29 años": 29,
                        "30 años": 30,
                        "31 años": 31,
                        "32 años": 32,
                        "33 años": 33,
                        "34 años": 34,
                        "35 años": 35,
                        "36 años": 36,
                        "37 años": 37,
                        "38 años": 38,
                        "39 años": 39,
                        "40 años": 40,
                        "41 años": 41,
                        "42 años": 42,
                        "43 años": 43,
                        "44 años": 44,
                        "45 años": 45,
                        "46 años": 46,
                        "47 años": 47,
                        "48 años": 48,
                        "49 años": 49,
                        "50 años": 50,
                        "51 años": 51,
                        "52 años": 52,
                        "53 años": 53,
                        "54 años": 54,
                        "55 años": 55,
                        "56 años": 56,
                        "57 años": 57,
                        "58 años": 58,
                        "59 años": 59,
                        "60 años": 60,
                        "61 años": 61,
                        "62 años": 62,
                        "63 años": 63,
                        "64 años": 64,
                        "65 años": 65,
                        "66 años": 66,
                        "67 años": 67,
                        "68 años": 68,
                        "69 años": 69,
                        "70 años": 70,
                        "71 años": 71,
                        "72 años": 72,
                        "73 años": 73,
                        "74 años": 74,
                        "75 años": 75}
    
    otros_planes_cuotas_opciones = {"banco":1,
                                    "tiendas":2,
                                    "ninguno":3}
    
    vivienda_opciones = {   "gratis":1,
                            "alquiler":2,
                            "propia":3}
    
    numero_creditos_opciones = {    "1":1,
                                    "2-3":2,
                                    "4-5":3,
                                    ">= 6":4}
    
    trabajo_opciones = {    "desempleado/no calificado - no residente":1,
                            "no calificado - residente":2,
                            "empleado calificado/funcionario":3,
                            "gerente/trabajador por cuenta propia/empleado altamente calificado":4}
    
    personas_dep_opciones = {"3 o más":1,"entre 0 y 2":2}
    
    telefono_opciones = {"no":1, "si (a nombre del cliente)":2}
    
    trabajador_extranjero_opciones = {"si":1,"no":2}

    # Lecctura de datos
    seleccion0 = st.text_input("Ingrese el nombre del solicitante:")
    nombreCompleto = seleccion0.title()   
    seleccion1 = st.selectbox("Seleccione el estado de la cuenta del solicitante:", list(estado_opciones.keys()))
    valor_estado = estado_opciones[seleccion1]
    seleccion2 = st.selectbox("Seleccione la duración del crédito en meses:", list(duracion_opciones.keys()))
    valor_duracion = duracion_opciones[seleccion2]
    seleccion3 = st.selectbox("Seleccione el historial del crédito del solicitante:", list(historial_cred_opciones.keys()))
    valor_historial_cred = historial_cred_opciones[seleccion3]
    seleccion4 = st.selectbox("Seleccione el propósito del crédito:", list(proposito_opciones.keys()))
    valor_proposito = proposito_opciones[seleccion4]    
    seleccion5 = st.text_input("Ingrese el monto del crédito:")
    valor_monto = seleccion5.title()    
    seleccion6 = st.selectbox("Seleccione los ahorros del solicitante:", list(ahorros_opciones.keys()))
    valor_ahorros = ahorros_opciones[seleccion6]    
    seleccion7 = st.selectbox("Seleccione la duración del empleo actual:", list(duracion_empleo_opciones.keys()))
    valor_duracion_emp = duracion_empleo_opciones[seleccion7]
    seleccion8 = st.selectbox("Seleccione la tasa de pago de cuotas del crédito:", list(tasa_pago_cuotas_opciones.keys()))
    valor_tasa_pago_cuotas = tasa_pago_cuotas_opciones[seleccion8]
    seleccion9 = st.selectbox("Seleccione el genero y estado civil del solicitante:", list(estado_personal_sex_opciones.keys()))
    valor_estado_personal_sex = estado_personal_sex_opciones[seleccion9]
    seleccion10 = st.selectbox("Seleccione si hay deudores o garantes para el crédito:", list(otros_deudores_opciones.keys()))
    valor_otros_deudores = otros_deudores_opciones[seleccion10]
    seleccion11 = st.selectbox("Seleccione los años que ha vivido en la residenci actual:", list(residencia_actual_opciones.keys()))
    valor_residencia_actual = residencia_actual_opciones[seleccion11]
    seleccion12 = st.selectbox("Seleccione la propiedad mas valiosa del solicitante:", list(propiedad_opciones.keys()))
    valor_propiedad = propiedad_opciones[seleccion12]
    seleccion13 = st.selectbox("Seleccione la edad del solicitante:", list(edad_opciones.keys()))
    valor_edad = edad_opciones[seleccion13]
    seleccion14 = st.selectbox("Seleccione con quien tiene otros planes de cuotas:", list(otros_planes_cuotas_opciones.keys()))
    valor_otros_planes_cuotas = otros_planes_cuotas_opciones[seleccion14]
    seleccion15 = st.selectbox("Seleccione el el tipo de vivienda actual del solicitante:", list(vivienda_opciones.keys()))
    valor_vivienda = vivienda_opciones[seleccion15]
    seleccion16 = st.selectbox("Seleccione el número de creditos que el solicitante tiene en este banco:", list(numero_creditos_opciones.keys()))
    valor_numero_creditos = numero_creditos_opciones[seleccion16]
    seleccion17 = st.selectbox("Seleccione el estado laboral del solicitante:", list(trabajo_opciones.keys()))
    valor_trabajo = trabajo_opciones[seleccion17]
    seleccion18 = st.selectbox("Seleccione el numero de personas que dependen financieramente del solicitante:", list(personas_dep_opciones.keys()))
    valor_personas_dep = personas_dep_opciones[seleccion18]
    seleccion19 = st.selectbox("Seleccione si hay una linea telefonica registrada a nombre del solicitante:", list(telefono_opciones.keys()))
    valor_telefono = telefono_opciones[seleccion19]
    seleccion20 = st.selectbox("Seleccione si el solicitante es un trabajador extranjero:", list(trabajador_extranjero_opciones.keys()))
    valor_trabajador_extranjero = trabajador_extranjero_opciones[seleccion20]
    
    #CREAR CONJUNTO DATOS
    DatosCrudos = pd.DataFrame({
            'laufkont': [valor_estado],
            'laufzeit': [valor_duracion],
            'moral': [valor_historial_cred],
            'verw': [valor_proposito],
            'hoehe': [valor_monto],
            'sparkont': [valor_ahorros],
            'beszeit': [valor_duracion_emp],
            'rate': [valor_tasa_pago_cuotas],
            'famges': [valor_estado_personal_sex],
            'buerge': [valor_otros_deudores],
            'wohnzeit': [valor_residencia_actual],
            'verm': [valor_propiedad],
            'alter': [valor_edad],
            'weitkred': [valor_otros_planes_cuotas],
            'wohn': [valor_vivienda],
            'bishkred': [valor_numero_creditos],
            'beruf': [valor_trabajo],
            'pers': [valor_personas_dep],
            'telef': [valor_telefono],
            'gastarb': [valor_trabajador_extranjero]
    })
    
    with st.container():
        st.write("---")   
        st.write("""
                 Creado por Mayron Rodríguez Sibaja 113250371 como parte del trabajo fin de master para la Universidad 
                 Complutense de Madrid para el Master Big data & Business Analytics modalidad Online(2022-2023).
                 El resultado del siguiente boton "Predecir" ejecuta un modelo de machine learning pre-entrenado 
                 con los datos ingresados anteriormente y realiza una predicción que permite saber 
                 si un solicitante de crédito va cumplir con sus obligaciones(good) ó
                 si va tener problemas con sus obligaciones(bad)
                 
                 """)  
        # El botón predicción se usa para iniciar el procesamiento
        if st.button("Predecir"):  
           predictS = model_prediction(DatosCrudos, Clasificador)
           predictSProba = model_prediction_prob(DatosCrudos, Clasificador)  
           predictSProba = int(round(predictSProba[0],2)*100)
           if predictSProba < 50:
              predictSProba = 100 - predictSProba
               
           if predictS[0] == 'good':
               st.success(f"El solicitante de crédito '{nombreCompleto}' se clasifica como: '{predictS[0]}' con una probabilidad de '{predictSProba}' por ciento.")
           else:
               st.error(f"El solicitante de crédito '{nombreCompleto}' se clasifica como: '{predictS[0]}' con una probabilidad de '{predictSProba}' por ciento.")
        
     

if __name__ == '__main__':
    main()
