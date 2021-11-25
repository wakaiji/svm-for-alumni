from flask import Flask, flash, redirect, url_for, render_template, request
from werkzeug.utils import secure_filename
import pickle
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split

import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo


ALLOWED_EXTENSIONS = set(['xlsx','csv'])

secret_key = os.urandom(12)

DETECT_FOLDER = 'static'
app = Flask(__name__)
app.secret_key = secret_key
app.config['UPLOAD_FOLDER'] = DETECT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024

def prediction_alumnus(file_path):
    splitfile = file_path.split(".")
    if splitfile[-1] == "xlsx":
        data_raw = pd.read_excel(file_path)
    else:
        data_raw = pd.read_csv(file_path)

    data_raw.columns = ["no", "fakultas", "program_studi", "tahun_lulus", "nama", "jangka_kerja", "jangka_belum_kerja", "jenis_perusahaan", "nama_perusahaan", "tingkatan"]
    
    data_raw = data_raw.dropna(subset=["jangka_kerja","jenis_perusahaan"])

    list_data = {"jenis_perusahaan": data_raw["jenis_perusahaan"], "jangka_kerja":data_raw["jangka_kerja"], "jangka_belum_kerja":data_raw["jangka_belum_kerja"],"tingkatan":data_raw["tingkatan"]}

    new_data = pd.DataFrame(list_data)
    new_data["jenis_perusahaan"] = new_data["jenis_perusahaan"].replace(["Belum Ada Pekerjaan","[1] Instansi Pemerintah","[2] Organisasi Non-profit/Lembaga Swadaya Masyarakat","[3] Perusahaan Swasta","[4] Wiraswasta/perusahaan Sendiri","[5] Lainnya","[6] BUMN/BUMD","[7] Institusi/Organisasi Multilateral"],[0,1,2,3,4,5,6,7])
    new_data["tingkatan"] = new_data["tingkatan"].replace(["-","[1] Setingkat Lebih Tinggi","[2] Tingkat Yang Sama","[3] Setingkat Lebih Rendah","[4] Tidak Perlu Pendidikan Tinggi"],[0,1,2,3,4])

    nama_perusahaan = list(data_raw["nama_perusahaan"])

    le = LabelEncoder()
    le.fit(nama_perusahaan)
    le.classes_
    perusahaan_le = le.transform(nama_perusahaan)

    new_data["perusahaan"] = perusahaan_le.tolist()

    x_test = pd.DataFrame(np.c_[new_data['jenis_perusahaan'],new_data['tingkatan'],new_data['jangka_kerja'],new_data['jangka_belum_kerja'],new_data['perusahaan']])
    model_filename = "pekerjaan_model.sav"
    loaded_model = pickle.load(open(model_filename, 'rb'))
    result = loaded_model.predict(x_test)
    
    data_raw["target"] = result
    data_raw["target"] = data_raw["target"].replace([0,1,2], ["Tidak Sesuai", "Kurang Sesuai", "Sesuai"])

    Visualize_class(data_raw, feature="target", title="Kesesuaian bidang kerja", filename="bar1.png")
    Visualize_class(data_raw, feature="jenis_perusahaan", title="Pembagian jenis perusahaan", filename="bar2.png")

    array_prediction = data_raw.to_numpy().tolist()

    return array_prediction

def training_alumnus(file_path):
    splitfile = file_path.split(".")
    if splitfile[-1] == "xlsx":
        data_raw = pd.read_excel(file_path)
    else:
        data_raw = pd.read_csv(file_path)

    data_raw.columns = ["no", "fakultas", "program_studi", "tahun_lulus", "nama", "jangka_kerja", "jangka_belum_kerja", "jenis_perusahaan", "nama_perusahaan", "tingkatan", "target"]

    data_raw["jenis_perusahaan"] = data_raw["jenis_perusahaan"].replace([np.nan],["Belum Ada Pekerjaan"])
    data_raw["nama_perusahaan"] = data_raw["nama_perusahaan"].replace([np.nan],["-"])
    data_raw["tingkatan"] = data_raw["tingkatan"].replace([np.nan],["-"])
    data_raw["jangka_kerja"] = data_raw['jangka_kerja'].replace([np.nan],[0])
    data_raw["jangka_belum_kerja"] = data_raw["jangka_belum_kerja"].replace([np.nan],[0])

    list_data = {"jenis_perusahaan": data_raw["jenis_perusahaan"], "jangka_kerja":data_raw["jangka_kerja"], "jangka_belum_kerja":data_raw["jangka_belum_kerja"],"tingkatan":data_raw["tingkatan"],"target":data_raw["target"]}

    new_data = pd.DataFrame(list_data)
    new_data["jenis_perusahaan"] = new_data["jenis_perusahaan"].replace(["Belum Ada Pekerjaan","[1] Instansi Pemerintah","[2] Organisasi Non-profit/Lembaga Swadaya Masyarakat","[3] Perusahaan Swasta","[4] Wiraswasta/perusahaan Sendiri","[5] Lainnya","[6] BUMN/BUMD","[7] Institusi/Organisasi Multilateral"],[0,1,2,3,4,5,6,7])
    new_data["tingkatan"] = new_data["tingkatan"].replace(["-","[1] Setingkat Lebih Tinggi","[2] Tingkat Yang Sama","[3] Setingkat Lebih Rendah","[4] Tidak Perlu Pendidikan Tinggi"],[0,1,2,3,4])
    new_data["target"] = new_data["target"].replace(["Tidak Sesuai", "Kurang Sesuai", "Sesuai"], [0,1,2])

    nama_perusahaan = list(data_raw["nama_perusahaan"])

    le = LabelEncoder()
    le.fit(nama_perusahaan)
    le.classes_
    perusahaan_le = le.transform(nama_perusahaan)

    new_data["perusahaan"] = perusahaan_le.tolist()

    X = pd.DataFrame(np.c_[new_data['jenis_perusahaan'],new_data['tingkatan'],new_data['jangka_kerja'],new_data['jangka_belum_kerja'],new_data['perusahaan']])
    Y = new_data['target']

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=True)
    linear_model = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo').fit(X_train, y_train)
    linear_pred = linear_model.predict(X_test)
    accuracy_linear = linear_model.score(X_test, y_test)

    cm_linear = confusion_matrix(y_test, linear_pred)

    save_confusion(cm_linear)
    y_pred = np.around(linear_pred)

    f1 = f1_score(y_test, y_pred,average=None).tolist()
    precision = precision_score(y_test, y_pred,average=None).tolist()
    recall = recall_score(y_test, y_pred,average=None).tolist()

    metrik = [f1,precision,recall]

    klasifikasi = classification_report(y_test, y_pred)

    return accuracy_linear, klasifikasi, metrik

def save_confusion(cm):
    plt.subplots(figsize=(15,8))
    sns.heatmap(data=cm, center=0, annot=True)
    plt.savefig("./static/plot/{}".format("confusion_matrix.png"))

def Visualize_class(df, feature, title, filename):
    num_image = df[feature].value_counts().rename_axis(feature).reset_index(name="jenis")
    fig = px.bar(num_image[::1], x="jenis", y=feature, orientation='h', color='jenis')
    fig.update_layout(
        title={
            'text' : title,
            'y' : 0.95,
            'x' : 0.5,
            'xanchor' : 'center',
            'yanchor' : 'top'})
    fig.write_image("./static/plot/{}".format(filename))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/training")
def training():
    return render_template("training.html")

@app.route("/training", methods=['POST'])
def upload_training_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_folder = os.path.join('static','file')
            file.save(os.path.join(save_folder, filename))
            # flash('file successfully uploaded')

            return redirect(url_for('training_result', filename=filename))
    else:
        flash('Allowed file types are excel')
        return redirect(request.url)

@app.route("/training_result/<filename>")
def training_result(filename):
    file_path = "./static/file/"
    excel_path = file_path + filename
    training, klasifikasi, metrik = training_alumnus(excel_path)
    kelas = ['Tidak Sesuai', 'Kurang Sesuai', 'Sesuai']
    return render_template("training_result.html", training = training, klasifikasi = klasifikasi, metrik=metrik, kelas=kelas)

@app.route("/testing")
def testing():
    return render_template("testing.html")

@app.route("/testing", methods=['POST'])
def upload_testing_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_folder = os.path.join('static','file')
            file.save(os.path.join(save_folder, filename))
            # flash('file successfully uploaded')

            return redirect(url_for('testing_result', filename=filename))
    else:
        flash('Allowed file types are excel')
        return redirect(request.url)

@app.route("/testing_result/<filename>")
def testing_result(filename):
    file_path = "./static/file/"
    excel_path = file_path + filename
    prediction_result = prediction_alumnus(excel_path)
    return render_template("testing_result.html", prediction = prediction_result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
