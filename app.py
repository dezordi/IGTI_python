import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import MinMaxScaler
normaliza = MinMaxScaler()


app = Flask(__name__)

def previsao_diabetes(lista_valores_formulario):
    prever = normaliza.fit_transform(np.array(lista_valores_formulario).reshape(8,1))
    prever = prever.reshape(1,8)
    print(prever)
    modelo_salvo = joblib.load('melhor_modelo.sav')
    resultado = modelo_salvo.predict(prever)
    return resultado[0]

@app.route('/')
def home():
    return render_template("index.html")


@app.route("/result",methods=['POST'])
def result():
    if request.method=='POST':
        lista_formulario = request.form.to_dict()
        print(lista_formulario)
        lista_formulario = list(lista_formulario.values())
        print(lista_formulario)
        lista_formulario = list(map(float,lista_formulario))
        print(type(lista_formulario))
        resultado = previsao_diabetes(lista_formulario)
        if int(resultado) == 1:
            previsao = "Possui diabetes"
        else:
            previsao = "Não possui diabetes"

        #retorna o valor
        print(previsao)
        return render_template("resultado.html", previsao = previsao)

if __name__ == "__main__":
    app.run(debug=True)