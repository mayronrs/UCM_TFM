#------------------------------
#Autor: Mayron Rodriguez Sibaja. 113250371
#Fecha: Setiembre, 2023
#Descripcion: Funciones para el pipeline de preprocesamiento de datos para un modelo de clasificacion de riesgo de credito.
#Para la Universidad Complutense de Madrid por el trabajo de fin de Master big data & business analytics
#Modalidad: Online (2022-2023)
#------------------------------

from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import OneHotEncoder,PowerTransformer, OrdinalEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd

class TraducirVariables(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        new_X = X.copy()
        new_X.columns = ["estado", "duracion", "historial_crediticio", "proposito", "monto", "ahorros", "duracion_empleo",
                 "tasa_cuota", "estado_personal_sexual", "otros_deudores", "residencia_actual", "propiedad", "edad",
                 "otros_planes_de_cuotas", "vivienda", "numero_creditos", "trabajo", "personas_responsables",
                 "telefono", "trabajador_extranjero"]
        return new_X
    
class RecodificarVariables(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):              
        #Definir niveles
        lvestado = {1:"no checking account", 2:"less than 0 DM", 3:"0 to 200 DM", 4:"200 DM or more"}

        lvhistorial_crediticio = {0 : "delay in paying off in the past",       
         1 : "critical account/other credits elsewhere",   
         2 : "no credits taken/all credits paid back duly",
         3 : "existing credits paid back duly till now",
         4 : "all credits at this bank paid back duly"}

        lvproposito = {0 : "others", 1 : "car (new)", 2 : "car (used)",3 : "furniture/equipment", 4 : "radio/television", 
                   5 : "domestic appliances", 6 : "repairs", 7 : "education", 8 : "vacation", 9 : "retraining", 10 : "business"}

        lvahorros = {1:"unknown/no savings account", 2 :"less than 100 DM", 3:"100 to 500 DM", 4 :"500 to 1000 DM", 5:"1000 DM or more"}
        lvduracion_empleo = {1 : "unemployed",2 : "less than 1 year", 3 : "1 to 4 yrs", 4 : "4 to 7 yrs", 5 : "7 yrs or more"}
        lvtasa_cuota = {1 : "35 or more", 2 : "25 to 35", 3 : "20 to 25", 4 : "less than 20"}
        lvestado_personal_sexual = {1 : "male : divorced/separated", 2 : "female : non-single or male : single", 3 : "male : married/widowed", 4 : "female : single"}

        lvotros_deudores = {1 : "none", 2 : "co-applicant", 3 : "guarantor"}
        lvresidencia_actual = {1 : "less than 1 year", 2 : "1 to 4 yrs", 3 : "4 to 7 yrs", 4 : "7 yrs or more"}
        lvpropiedad = {1 : "unknown/no property", 2 : "car or other",3 : "building soc. savings agr./life insurance", 4 : "real estate"}
        lvotros_planes_de_cuotas = {1 : "bank",2 : "stores",3 : "none"}
        lvvivienda = {1:"for free", 2:"rent", 3:"own"} 
        lvnumero_creditos = {1 : "1",2 : "2-3", 3 : "4-5",4 : "6 or more"}
        lvtrabajo = {1 : "unemployed/unskilled - non-resident",2 : "unskilled-resident",3 : "skilled employee/official",4 : "manager/self-employed/highly qualified employee"}

        lvpersonas_responsables = {1 : "3 or more", 2 : "0 to 2"}
        lvtelefono = {1 : "no",2 : "yes (under customer name)"}
        lvtrabajador_extranjero = {1 : "yes", 2 : "no"}
        
        #Mapear los niveles
        X["estado"] = X["estado"].map(lvestado)
        X["historial_crediticio"] = X["historial_crediticio"].map(lvhistorial_crediticio)
        X["proposito"] = X["proposito"].map(lvproposito)
        X["ahorros"] = X["ahorros"].map(lvahorros)
        X["duracion_empleo"] = X["duracion_empleo"].map(lvduracion_empleo)
        X["tasa_cuota"] = X["tasa_cuota"].map(lvtasa_cuota)
        X["estado_personal_sexual"] = X["estado_personal_sexual"].map(lvestado_personal_sexual)
        X["otros_deudores"] = X["otros_deudores"].map(lvotros_deudores)
        X["residencia_actual"] = X["residencia_actual"].map(lvresidencia_actual)
        X["propiedad"] = X["propiedad"].map(lvpropiedad)
        X["otros_planes_de_cuotas"] = X["otros_planes_de_cuotas"].map(lvotros_planes_de_cuotas)
        X["vivienda"] = X["vivienda"].map(lvvivienda)
        X["numero_creditos"] = X["numero_creditos"].map(lvnumero_creditos)
        X["trabajo"] = X["trabajo"].map(lvtrabajo)
        X["personas_responsables"] = X["personas_responsables"].map(lvpersonas_responsables)
        X["telefono"] = X["telefono"].map(lvtelefono)
        X["trabajador_extranjero"] = X["trabajador_extranjero"].map(lvtrabajador_extranjero)
        
        return X
    
class ImputacionVariables(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        #ORDINALES
        X["numero_creditos"] = X["numero_creditos"].replace(["4-5", "6 or more"], "4 or more")
        X["ahorros"] = X["ahorros"].replace(["100 to 500 DM", "500 to 1000 DM"], "100 to 1000 DM")
       
        #NOMINALES
        X["trabajo"] = X["trabajo"].replace(["unskilled-resident", "unemployed/unskilled - non-resident"], "unskilled-resident or unemployed/unskilled non-resident")
        X["proposito"] = X["proposito"].replace(["domestic appliances", "radio/television","repairs"], "domestic appliances or radio/television or repairs")
        X["proposito"] = X["proposito"].replace(["retraining","business", "vacation"], "business or vacation or retraining")
        X["proposito"] = X["proposito"].replace(["car (new)", "car (used)"], "car (new) or car (used)")
        
        return X
    
class TratarVariablesCatOrdinales(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        estado_OrEncoder = OrdinalEncoder(categories=[("no checking account", "less than 0 DM", "0 to 200 DM", "200 DM or more")])
        ahorros_OrEncoder = OrdinalEncoder(categories=[("unknown/no savings account", "less than 100 DM", "100 to 1000 DM", "1000 DM or more")])
        dur_empleo_OrEncoder = OrdinalEncoder(categories=[("unemployed", "less than 1 year", "1 to 4 yrs", "4 to 7 yrs", "7 yrs or more")])
        tasa_OrEncoder = OrdinalEncoder(categories=[("35 or more", "25 to 35", "20 to 25", "less than 20")])
        residencia_OrEncoder = OrdinalEncoder(categories=[("less than 1 year", "1 to 4 yrs", "4 to 7 yrs", "7 yrs or more")])
        #Se cambia el orden cosiderando que al tener mas creditos es mayor el riesgo de que no pueda pagar.
        num_creditos_OrEncoder = OrdinalEncoder(categories=[("4 or more","2-3","1")])
        #Se cambia el orden cosiderando que mientras menos personas dependan de el es mejor
        per_responsables_OrEncoder = OrdinalEncoder(categories=[("3 or more", "0 to 2")])
        
        col_transformer = ColumnTransformer(
        transformers=[
            ('estado_OrEncoder', estado_OrEncoder, ['estado']),
            ('ahorros_OrEncoder', ahorros_OrEncoder, ['ahorros']),
            ('dur_empleo_OrEncoder', dur_empleo_OrEncoder, ['duracion_empleo']),
            ('tasa_OrEncoder', tasa_OrEncoder, ['tasa_cuota']),
            ('residencia_OrEncoder', residencia_OrEncoder, ['residencia_actual']),
            ('num_creditos_OrEncoder', num_creditos_OrEncoder, ['numero_creditos']),
            ('per_responsables_OrEncoder', per_responsables_OrEncoder, ['personas_responsables']),

            ],remainder='passthrough')
        
        Datatransformed = col_transformer.fit_transform(X)
            
        NombreColumnas = ['estado','ahorros','duracion_empleo', 'tasa_cuota','residencia_actual','numero_creditos','personas_responsables',
                  'duracion', 'historial_crediticio', 'proposito', 'monto',
                  'estado_personal_sexual', 'otros_deudores',  'propiedad', 'edad',
                  'otros_planes_de_cuotas', 'vivienda',  'trabajo',
                  'telefono', 'trabajador_extranjero']
            
        X =pd.DataFrame(Datatransformed,columns = NombreColumnas)
        X['estado'] = X['estado'].astype(int)
        X['ahorros'] = X['ahorros'].astype(int)
        X['duracion_empleo'] = X['duracion_empleo'].astype(int)
        X['tasa_cuota'] = X['tasa_cuota'].astype(int)
        X['residencia_actual'] = X['residencia_actual'].astype(int)
        X['numero_creditos'] = X['numero_creditos'].astype(int)
        X['personas_responsables'] = X['personas_responsables'].astype(int)

        X['duracion'] = X['duracion'].astype(float)
        X['monto'] = X['monto'].astype(float)
        X['edad'] = X['edad'].astype(float)
        
        return X
    
class EscalarVariables(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        numericas_pt = PowerTransformer(method='box-cox')
        col_num = ['duracion', 'monto', 'edad']
        col_transformer2 = ColumnTransformer(
        transformers=[
            ('Outliers', numericas_pt, col_num)
            ],remainder='passthrough')
        
        Datatransformed = X
        
        if (X.shape[0] != 1):
             Datatransformed = col_transformer2.fit_transform(X)
        
        
        NombreColumnas = ['duracion', 'monto', 'edad',
                  'estado','ahorros','duracion_empleo', 'tasa_cuota','residencia_actual','numero_creditos','personas_responsables',
                  'historial_crediticio', 'proposito', 
                  'estado_personal_sexual', 'otros_deudores',  'propiedad', 
                  'otros_planes_de_cuotas', 'vivienda',  'trabajo',
                  'telefono', 'trabajador_extranjero']
        
        X =pd.DataFrame(Datatransformed,columns = NombreColumnas)
        X['estado'] = X['estado'].astype(int)
        X['ahorros'] = X['ahorros'].astype(int)
        X['duracion_empleo'] = X['duracion_empleo'].astype(int)
        X['tasa_cuota'] = X['tasa_cuota'].astype(int)
        X['residencia_actual'] = X['residencia_actual'].astype(int)
        X['numero_creditos'] = X['numero_creditos'].astype(int)
        X['personas_responsables'] = X['personas_responsables'].astype(int)

        X['duracion'] = X['duracion'].astype(float)
        X['monto'] = X['monto'].astype(float)
        X['edad'] = X['edad'].astype(float)
        
        return X
    
class DummyVariables(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        
        col_cat = X.select_dtypes(include="object").columns
        
        dummy_oh = OneHotEncoder(drop='if_binary', sparse=False)
        dummy_oh = dummy_oh.fit(X[col_cat])
        column_dummynames = dummy_oh.get_feature_names_out(col_cat)
        
        col_transformer3 = ColumnTransformer(
        transformers=[
            ('Dummies', dummy_oh, col_cat)
            ],remainder='passthrough')
        
        Datatransformed = col_transformer3.fit_transform(X)
        
        NombreColumnas = column_dummynames.tolist() + [
                  'duracion', 'monto', 'edad',
                  'estado','ahorros','duracion_empleo', 'tasa_cuota','residencia_actual','numero_creditos',
                  'personas_responsables']
        
        X =pd.DataFrame(Datatransformed,columns = NombreColumnas)
        X[column_dummynames.tolist()] = X[column_dummynames.tolist()].astype(int)      
        X['estado'] = X['estado'].astype(int)
        X['ahorros'] = X['ahorros'].astype(int)
        X['duracion_empleo'] = X['duracion_empleo'].astype(int)
        X['tasa_cuota'] = X['tasa_cuota'].astype(int)
        X['residencia_actual'] = X['residencia_actual'].astype(int)
        X['numero_creditos'] = X['numero_creditos'].astype(int)
        X['personas_responsables'] = X['personas_responsables'].astype(int)

        X['duracion'] = X['duracion'].astype(float)
        X['monto'] = X['monto'].astype(float)
        X['edad'] = X['edad'].astype(float)
        
        
        #Cambiar los nombres de la dummys
        nuevos_nombres = {
            'historial_crediticio_all credits at this bank paid back duly': 'historial_cred_all',
            'historial_crediticio_critical account/other credits elsewhere': 'historial_cred_critical',
            'historial_crediticio_delay in paying off in the past': 'historial_cred_delay',
            'historial_crediticio_existing credits paid back duly till now': 'historial_cred_existing',
            'historial_crediticio_no credits taken/all credits paid back duly': 'historial_cred_no_credits',
            'proposito_business or vacation or retraining': 'proposito_business_vacation',
            'proposito_car (new) or car (used)': 'proposito_car_new_used',
            'proposito_domestic appliances or radio/television or repairs': 'proposito_domestic_app_repairs',
            'proposito_furniture/equipment': 'proposito_furn_equipment',
            'proposito_others': 'proposito_others',
            'estado_personal_sexual_female : non-single or male : single': 'estado_per_sex_non_single_male',
            'estado_personal_sexual_female : single': 'estado_per_sex_single',
            'estado_personal_sexual_male : divorced/separated': 'estado_per_sex_divorced_separated',
            'estado_personal_sexual_male : married/widowed': 'estado_per_sex_married_widowed',
            'otros_deudores_co-applicant': 'otros_deud_co_applicant',
            'otros_deudores_guarantor': 'otros_deud_guarantor',
            'otros_deudores_none': 'otros_deud_none',
            'propiedad_building soc. savings agr./life insurance': 'propiedad_building',
            'propiedad_car or other': 'propiedad_car_other',
            'propiedad_real estate': 'propiedad_real_estate',
            'propiedad_unknown/no property': 'propiedad_unknown_no_property',
            'otros_planes_de_cuotas_bank': 'otros_planes_bank',
            'otros_planes_de_cuotas_none': 'otros_planes_none',
            'otros_planes_de_cuotas_stores': 'otros_planes_stores',
            'vivienda_for free': 'vivienda_free',
            'vivienda_own': 'vivienda_own',
            'vivienda_rent': 'vivienda_rent',
            'trabajo_manager/self-employed/highly qualified employee': 'trabajo_manager',
            'trabajo_skilled employee/official': 'trabajo_skilled_employee',
            'trabajo_unskilled-resident or unemployed/unskilled non-resident': 'trabajo_unski_resident_unemploy_non_resident',
            'telefono_yes (under customer name)': 'telefono_yes',
            'trabajador_extranjero_yes': 'trabajador_extranjero_yes' 


        }

        # Utilizar el método rename para cambiar los nombres de las columnas
        X.rename(columns=nuevos_nombres, inplace=True)        
        
        
        return X  

class SeleccionVariables(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
                
        #Cada vez que se reentrenen el modelo hay que actualizar estas variables seleccionadas en caso que cambien.
        mejoresVariables= ['historial_cred_all',
                             'historial_cred_critical',
                             'historial_cred_delay',
                             'proposito_business_vacation',
                             'proposito_car_new_used',
                             'proposito_domestic_app_repairs',
                             'proposito_furn_equipment',
                             'proposito_others',
                             'estado_per_sex_married_widowed',
                             'otros_deud_co_applicant',
                             'otros_deud_guarantor',
                             'propiedad_real_estate',
                             'propiedad_unknown_no_property',
                             'otros_planes_none',
                             'telefono_yes',
                             'trabajador_extranjero_yes',
                             'duracion',
                             'edad',
                             'estado',
                             'ahorros',
                             'personas_responsables']
        
        
        #En caso de que alguna columna no exista se agrega con 0
        for columna in mejoresVariables:
            if columna not in X.columns:
                X[columna] = 0
        
        X = X[mejoresVariables]
        
        return X
    