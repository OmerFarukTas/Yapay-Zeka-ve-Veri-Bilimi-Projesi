import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import graphviz
from sklearn.tree import export_graphviz

# Verileri yükle ve hazırlık
veriler = pd.read_csv('breast-cancer.csv')

#Hedef değişkenin olduğu dizi.
y = veriler['diagnosis']

#Korelasyon Matrisi yardımı ile 30 değişken arasından sonuca en çok etki eden 7 değişkeni aldık.
'''
le=preprocessing.LabelEncoder()
c = le.fit_transform(y)

c_dataframe = pd.DataFrame(data=c,index=range(569),columns=['tahmin'])

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
veriler2 = pd.concat([c_dataframe,x],axis=1)
print(veriler2.corr())'''

#Girdilerin olduğu dizi.
x = veriler.iloc[:, [2, 4, 5, 9, 24, 25, 29]]

#Verileri eğitim ve test olarak ayırır.
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=34)

#Karar Ağacı modelini burada oluşturduk.
tree = DecisionTreeClassifier()

#Fit fonksiyonu ile modeli eğittik.
tree.fit(x_train, y_train)

#Textboxlara sadece float değer girilmesini sağlar.
def float_validation(action, index, value_if_allowed, prior_value, text, validation_type, trigger_type, widget_name):
    if action == '1':  # Ekleme işlemi
        if text in '0123456789.-+eE':  # Geçerli karakterler
            try:
                float(value_if_allowed)  # Eğer float değeriyse, izin ver
                return True
            except ValueError:
                return False
        else:
            return False
    elif action == '0':  # Silme işlemi
        return True
    return False
#Kullanıcıdan değerlerin alındığı ve modele koyularak tahmin yaptıran fonksiyon.
def islem_yap():
    try:
        radius_mean = float(entries[0].get())
        perimeter_mean = float(entries[1].get())
        area_mean = float(entries[2].get())
        concave_points_mean = float(entries[3].get())
        perimeter_worst = float(entries[4].get())
        area_worst = float(entries[5].get())
        concave_points_worst = float(entries[6].get())
        
        # Kullanıcıdan alınan verileri bir DataFrame'e koy
        user_data = pd.DataFrame([[radius_mean, perimeter_mean, area_mean, concave_points_mean, perimeter_worst, area_worst, concave_points_worst]], 
                                 columns=["radius_mean", "perimeter_mean", "area_mean", "concave points_mean", "perimeter_worst", "area_worst", "concave points_worst"])
        
        # Tahmin yap
        prediction = tree.predict(user_data)[0]
        accuracy = tree.score(x_test, y_test)
        
        # Tahmin sonucunu string olarak al
        prediction_str = "İyi Huylu" if prediction == 'B' else "Kötü Huylu"
        
        result_text = f"Tahmin: {prediction_str}\nModel Doğruluğu: {accuracy:.2f}"
    except ValueError:
        messagebox.showerror("Geçersiz giriş", "Lütfen tüm alanlara geçerli bir sayı girin.")
        return
    
    result_label.config(text=result_text)

def karar_agaci():
    # Karar ağacını görselleştir
    dot_data = export_graphviz(tree, out_file=None, 
                               feature_names=x.columns,  
                               class_names=y.unique(),  
                               filled=True, rounded=True,  
                               special_characters=True)  

    graph = graphviz.Source(dot_data)  
    graph.render("karar_agaci")
    graph.view()

#Karmaşıklık matrisinin oluşturulduğu fonksiyon.
def karmaşıklık_matrisi_goster():
    #Tahmin yapılarak tahmine göre karmaşıklık matrisini oluşturur.
    y_pred = tree.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    
    # Karmaşıklık matrisini başlıklar ile göster
    cm_df = pd.DataFrame(cm, index=["İyi Huylu", "Kötü Huylu"], columns=["İyi Huylu", "Kötü Huylu"])
    
    # Yeni bir pencere oluştur
    cm_window = tk.Toplevel(root)
    cm_window.title("Karmaşıklık Matrisi")
    
    # Matris başlıklarını ekle
    tk.Label(cm_window, text="").grid(row=0, column=0)
    tk.Label(cm_window, text="İyi Huylu").grid(row=0, column=1)
    tk.Label(cm_window, text="Kötü Huylu").grid(row=0, column=2)
    
    # Matris satır başlıklarını ve verilerini ekle
    tk.Label(cm_window, text="İyi Huylu").grid(row=1, column=0)
    tk.Label(cm_window, text=str(cm_df.at["İyi Huylu", "İyi Huylu"])).grid(row=1, column=1)
    tk.Label(cm_window, text=str(cm_df.at["İyi Huylu", "Kötü Huylu"])).grid(row=1, column=2)
    
    tk.Label(cm_window, text="Kötü Huylu").grid(row=2, column=0)
    tk.Label(cm_window, text=str(cm_df.at["Kötü Huylu", "İyi Huylu"])).grid(row=2, column=1)
    tk.Label(cm_window, text=str(cm_df.at["Kötü Huylu", "Kötü Huylu"])).grid(row=2, column=2)

# Ana pencereyi oluştur
root = tk.Tk()
root.title("Karar Ağacı ile Tahmin Uygulaması")
root.geometry("400x500")  # Sabit boyut

# Pencerenin yeniden boyutlandırılmasını devre dışı bırak
root.resizable(False, False)

# Giriş alanları ve etiketler için bir çerçeve oluştur
frame_entries = tk.Frame(root)
frame_entries.pack(pady=20, padx=20)

# Doğrulama fonksiyonunu kaydet
vcmd = (root.register(float_validation), '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')

# Değişkenler için etiketler ve giriş kutuları
labels = ["Radius Mean", "Perimeter Mean", "Area Mean", "Concave Points Mean", "Perimeter Worst", "Area Worst", "Concave Points Worst"]
entries = []

#Değişkenlerin girilmesi için textboxlar ve labellar oluşturulur.
for i, label in enumerate(labels):
    lbl = tk.Label(frame_entries, text=label, anchor='w')
    lbl.grid(row=i, column=0, padx=5, pady=5, sticky="ew")

    entry = tk.Entry(frame_entries, validate='key', validatecommand=vcmd)
    entry.grid(row=i, column=1, padx=5, pady=5, sticky="ew")
    entries.append(entry)

# Butonlar için bir çerçeve oluştur
frame_buttons = tk.Frame(root)
frame_buttons.pack(pady=20)

# Butonları oluştur ve çerçeveye yerleştir
btn_width = 20
btn_islem = tk.Button(frame_buttons, text="Sonuc", command=islem_yap, width=btn_width)
btn_kararagaci = tk.Button(frame_buttons, text="Karar Ağacı Göster", command=karar_agaci, width=btn_width)
btn_karışıklık = tk.Button(frame_buttons, text="Karmaşıklık Matrisi Göster", command=karmaşıklık_matrisi_goster, width=btn_width)

btn_islem.pack(pady=5)
btn_kararagaci.pack(pady=5)
btn_karışıklık.pack(pady=5)

# Sonuç alanı
result_label = tk.Label(root, text="Sonucları Buradan Öğrenebilirsiniz", justify='left')
result_label.pack(pady=10)

# Pencereyi çalıştır
root.mainloop()
