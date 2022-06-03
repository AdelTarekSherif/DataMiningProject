from tkinter.messagebox import askyesno

import numpy
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame
import tkinter as tk
from tkinter import ttk, RIGHT, Y, NONE, END, TOP, X, filedialog
from sklearn import metrics, preprocessing, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

feature_cols = ['Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 'FBS over 120', 'EKG results',
                'Max HR', 'Exercise angina', 'ST depression', 'Slope of ST', 'Number of vessels fluro',
                'Thallium']
le = preprocessing.LabelEncoder()
flag1 = False
flag2 = False
selection1 = 1

form = tk.Tk()
form.config(bg="#EFE7E2")
form.geometry("800x700")
form.title('Data Mining')
form.resizable(0, 0)

# frame for the data
frame1 = tk.LabelFrame(form, text="Data", font=('helvetica', 10, 'bold'))
frame1.place(width=800, height=260)
frame2 = tk.LabelFrame(form, bd=0, text="Algorithms:", bg="#EFE7E2", font=('helvetica', 12, 'bold'))
frame2.place(width=800, height=140, relx=0, rely=0.45)  # frame for algorithms
frame3 = tk.LabelFrame(frame2, bd=0, bg="#EFE7E2")
frame3.place(width=370, height=90, relx=0.38, rely=0.2)
frame4 = tk.LabelFrame(form, bd=4, bg="white")
frame4.place(width=400, height=240, relx=0, rely=0.65)
frame6 = tk.LabelFrame(form, bd=0, bg="#EFE7E2")
frame6.place(width=800, height=30, relx=0, rely=0.38)
frame7 = tk.LabelFrame(form, bd=4, bg="white")
frame7.place(width=400, height=240, relx=0.5, rely=0.65)

algo_selection = tk.IntVar()
R1 = tk.Radiobutton(frame2, text="Decision Tree", variable=algo_selection, value=1, bg="#EFE7E2",
                    command=lambda: selection())
R1.pack(anchor=tk.W)
R1.select()
# anchor:It represents the exact position of the text within the widget
# if the widget contains more space than the requirement of the text. The default value is CENTER.

R2 = tk.Radiobutton(frame2, text="KNN", variable=algo_selection, value=2, bg="#EFE7E2",
                    command=lambda: selection()).pack(anchor=tk.W)
R3 = tk.Radiobutton(frame2, text="NB", variable=algo_selection, value=3, bg="#EFE7E2",
                    command=lambda: selection()).pack(anchor=tk.W)

R4 = tk.Radiobutton(frame2, text="Random Forest", variable=algo_selection, value=4, bg="#EFE7E2",
                    command=lambda: selection()).pack(anchor=tk.W)
preprocessing_selection = tk.IntVar()

SubmitButton = tk.Button(frame2, text="Submit", bg="#EEEFEF", bd=1, width=15, command=lambda: start())
SubmitButton.place(relx=0.15, rely=0.78)
button1 = tk.Button(frame6, text="Load data", bg="#1789E6", fg="black", width=15, command=lambda: file_dialog()).grid(
    row=0,
    column=0)
button2 = tk.Button(frame6, text="Clear", bg="red", fg="white", width=15, command=lambda: clear_data())
button2.place(relx=0.85, rely=0)
button3 = tk.Button(frame6, text="Shape", width=15, bg="#EEEFEF", command=lambda: shape()).grid(row=0, column=1)
button4 = tk.Button(frame6, text="Preprocessing", bg="#17E62C", fg="black", width=15,
                    command=lambda: preprocessing()).grid(row=0, column=4)
button6 = tk.Button(frame6, text="Sum of Nulls", bg="#EEEFEF", width=15, command=lambda: sumOfNulls()).grid(row=0,
                                                                                                            column=2)
button7 = tk.Button(frame6, text="Sum of Zeros", bg="#EEEFEF", width=15, command=lambda: sumOfZeros()).grid(row=0,
                                                                                                            column=3)

tv1 = ttk.Treeview(frame1)
tv1.place(relheight=1, relwidth=1)

scrollX = tk.Scrollbar(frame1, orient="horizontal", command=tv1.xview)
scrollY = tk.Scrollbar(frame1, orient="vertical", command=tv1.yview)
tv1.configure(xscrollcommand=scrollX.set, yscrollcommand=scrollY.set)
scrollX.pack(side="bottom", fill="x")
scrollY.pack(side="right", fill="y")


def file_dialog():
    global dataset
    global read_file
    try:
        filename = filedialog.askopenfilename()
        read_file = pd.read_csv(filename)
        dataset = DataFrame(read_file)
    except ValueError:
        tk.messagebox.showerror("Information", "The file you have chosen is invalid")
        return None
    except FileNotFoundError:
        tk.messagebox.showerror("Information", "No file found")
        return None
    tv1.delete(*tv1.get_children())
    tv1["column"] = list(dataset.columns)
    tv1["show"] = "headings"
    for column in tv1["column"]:
        tv1.heading(column, text=column)
    dataset_rows = dataset.to_numpy().tolist()
    for row in dataset_rows:
        tv1.insert("", "end", values=row)
    global flag1
    flag1 = True


def load_data():
    tv1.delete(*tv1.get_children())
    tv1["column"] = list(dataset.columns)
    tv1["show"] = "headings"
    for column in tv1["column"]:
        tv1.heading(column, text=column)
    dataset_rows = dataset.to_numpy().tolist()
    for row in dataset_rows:
        tv1.insert("", "end", values=row)
    global flag1
    flag1 = True


def clear_data():
    tv1["column"] = list()
    tv1.delete(*tv1.get_children())
    for widget in frame4.winfo_children():
        widget.destroy()
    for widget in frame7.winfo_children():
        widget.destroy()
    global flag1
    flag1 = False
    global flag2
    flag2 = False


def selection():
    global selection1
    selection1 = int(algo_selection.get())
    global number1
    number1 = tk.IntVar()
    entry1 = tk.Entry(frame3, textvariable=number1, bd=5)
    if selection1 == 2:
        tk.Label(frame3, text="Number of neighbours:", font=('helvetica', 10, 'bold')).grid(row=0, column=0)
        entry1.grid(row=0, column=1)
    else:
        for widget in frame3.winfo_children():
            widget.destroy()


def start():
    for widget in frame4.winfo_children():
        widget.destroy()
    for widget in frame7.winfo_children():
        widget.destroy()
    if selection1 == 1:
        DecisionTree()
    elif selection1 == 2:
        global nn
        nn = number1.get()
        KNN()
    elif selection1 == 3:
        NB()
    elif selection1 == 4:
        RF()


def preprocessing():
    for widget in frame4.winfo_children():
        widget.destroy()
    if flag1:
        R5 = tk.Radiobutton(frame7, text="Transform into numeric values", bg="white", variable=preprocessing_selection,
                            value=1,
                            command=lambda: selection())
        R5.select()
        R6 = tk.Radiobutton(frame7, text="Convert zeros to mean", bg="white", variable=preprocessing_selection, value=2,
                            command=lambda: selection())
        R7 = tk.Radiobutton(frame7, text="Convert null to mean", bg="white", variable=preprocessing_selection, value=3,
                            command=lambda: selection())
        R8 = tk.Radiobutton(frame7, text="Convert zeros to median", bg="white", variable=preprocessing_selection,
                            value=4,
                            command=lambda: selection())
        R9 = tk.Radiobutton(frame7, text="Convert null to median", bg="white", variable=preprocessing_selection,
                            value=5,
                            command=lambda: selection())
        global index
        index = tk.IntVar()
        entry1 = tk.Entry(frame4, textvariable=index, bd=5)
        tk.Label(frame4, text="Enter index of column:", bg="white", font=('helvetica', 10, 'bold')).grid(column=0,
                                                                                                         row=0)
        entry1.grid(column=1, row=0)
        button5 = tk.Button(frame7, text="Start Preprocessing", bg="#17E62C", fg="black", width=15,
                            command=lambda: startPP())
        button5.place(relx=0.35, rely=0.8)
        R5.pack(anchor=tk.W)
        R6.pack(anchor=tk.W)
        R7.pack(anchor=tk.W)
        R8.pack(anchor=tk.W)
        R9.pack(anchor=tk.W)
    else:
        tk.Label(frame4, text="Please load the data", bg="white", font=('helvetica', 10, 'bold')).grid(column=0, row=0)


def startPP():
    global colID
    colID = index.get()
    selection2 = int(preprocessing_selection.get())
    for widget in frame4.winfo_children():
        widget.destroy()
    if selection2 == 1:
        transform()
    elif selection2 == 2:
        convertToMean(0)
    elif selection2 == 3:
        convertToMean(1)
    elif selection2 == 4:
        convertToMedian(0)
    elif selection2 == 5:
        convertToMedian(1)
    load_data()


def transform():
    colname = dataset.columns[colID]
    dataset[colname] = le.fit_transform(dataset[colname])
    global flag2
    flag2 = True
    label = tk.Label(frame4, text="The data was preprocessed successfully", bg="white", font=('helvetica', 10, 'bold'))
    label.place(relx=0.2, rely=0.4)
    for widget in frame7.winfo_children():
        widget.destroy()


def convertToMean(x):
    colname = dataset.columns[colID]
    if x == 0:
        dataset[colname] = dataset[colname].replace(0, numpy.NaN)
        dataset.fillna(dataset.mean(), inplace=True)
    else:
        dataset.fillna(dataset.mean(), inplace=True)
    global flag2
    flag2 = True
    label = tk.Label(frame4, text="The data was preprocessed successfully", bg="white", font=('helvetica', 10, 'bold'))
    label.place(relx=0.3, rely=0.4)
    for widget in frame7.winfo_children():
        widget.destroy()


def convertToMedian(x):
    colname = dataset.columns[colID]
    if x == 0:
        dataset[colname] = dataset[colname].replace(0, numpy.NaN)
    dataset.fillna(dataset.median(), inplace=True)
    global flag2
    flag2 = True
    label = tk.Label(frame4, text="The data was preprocessed successfully", bg="white", font=('helvetica', 10, 'bold'))
    label.place(relx=0.3, rely=0.4)
    for widget in frame7.winfo_children():
        widget.destroy()


def shape():
    for widget in frame4.winfo_children():
        widget.destroy()
    for widget in frame7.winfo_children():
        widget.destroy()
    if flag1:
        shapeofdataset = dataset.shape
        tk.Label(frame4, text="Shape:", bg="white", font=('helvetica', 10, 'bold')).grid(column=0, row=0)
        tk.Label(frame4, text=shapeofdataset, bg="white", font=('helvetica', 10, 'bold')).grid(column=1, row=0)
    else:
        tk.Label(frame4, text="Please load the data", bg="white", font=('helvetica', 10, 'bold')).grid(column=0, row=0)


def sumOfNulls():
    for widget in frame4.winfo_children():
        widget.destroy()
    for widget in frame7.winfo_children():
        widget.destroy()
    if flag1:
        sumofnulls = dataset.isnull().sum()
        v = tk.Scrollbar(frame4)
        v.pack(side=RIGHT, fill=Y)
        t = tk.Text(frame4, width=15, height=15, wrap=NONE,
                    yscrollcommand=v.set)

        t.insert(END, sumofnulls)
        t.pack(side=TOP, fill=X)
        v.config(command=t.xview)
    else:
        tk.Label(frame4, text="Please load the data", bg="white", font=('helvetica', 10, 'bold')).grid(column=0, row=0)


def sumOfZeros():
    for widget in frame4.winfo_children():
        widget.destroy()
    for widget in frame7.winfo_children():
        widget.destroy()
    if flag1:
        sumofzeros = (dataset == 0).sum()
        v = tk.Scrollbar(frame4)
        v.pack(side=RIGHT, fill=Y)
        t = tk.Text(frame4, width=15, height=15, wrap=NONE,
                    yscrollcommand=v.set)
        t.insert(END, sumofzeros)
        t.pack(side=TOP, fill=X)
        v.config(command=t.xview)
    else:
        tk.Label(frame4, text="Please load the data", bg="white", font=('helvetica', 10, 'bold')).grid(column=0, row=0)


def confirm(x):
    answer = askyesno(title='Confirmation',
                      message='Are you sure that you want to continue without preprocessing?')
    global flag2
    if answer:
        flag2 = True
        if x == 1:
            DecisionTree()
        elif x == 2:
            KNN()
        elif x == 3:
            NB()
        elif x == 4:
            RF()


def DecisionTree():
    for widget in frame4.winfo_children():
        widget.destroy()
    if flag1:
        if flag2:
            x = dataset[feature_cols]
            y = dataset['Heart Disease']
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
            global clf
            clf = DecisionTreeClassifier()
            clf = clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            accuracy = metrics.accuracy_score(y_test, y_pred) * 100
            tk.Label(frame4, text="Accuracy of Decision Tree algorithm:", bg="white",
                     font=('helvetica', 10, 'bold')).grid(column=0, row=0)
            tk.Label(frame4, text=accuracy, bg="white", font=('helvetica', 10, 'bold')).grid(column=1, row=0)

            tk.Button(frame4, text="Show tree", width=15, bg='#18F068', fg='black',
                      command=lambda: showTree('Decision')).place(relx=0.35, rely=0.8)

        else:
            confirm(1)

    else:
        tk.Label(frame4, text="Please load the data", bg="white", font=('helvetica', 10, 'bold')).grid(column=0, row=0)


def KNN():
    for widget in frame4.winfo_children():
        widget.destroy()
    if flag1:
        if flag2:
            x = dataset[feature_cols]
            y = dataset['Heart Disease']
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
            knn = KNeighborsClassifier(n_neighbors=nn)
            knn = knn.fit(x_train, y_train)
            y_pred = knn.predict(x_test)
            accuracy = metrics.accuracy_score(y_test, y_pred) * 100
            tk.Label(frame4, text="Accuracy of KNN algorithm:", bg="white", font=('helvetica', 10, 'bold')).grid(
                column=0, row=2)
            tk.Label(frame4, text=accuracy, bg="white", font=('helvetica', 10, 'bold')).grid(column=1, row=2)
        else:
            confirm(2)
    else:
        tk.Label(frame4, text="Please load the data", bg="white", font=('helvetica', 10, 'bold')).grid(column=0, row=0)


def NB():
    for widget in frame4.winfo_children():
        widget.destroy()
    if flag1:
        if flag2:
            x = dataset[feature_cols]
            y = dataset['Heart Disease']
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
            gnb = GaussianNB()
            gnb.fit(x_train, y_train)
            y_pred = gnb.predict(x_test)
            accuracy = metrics.accuracy_score(y_test, y_pred) * 100
            tk.Label(frame4, text="Accuracy of NB algorithm:", bg="white", font=('helvetica', 10, 'bold')).grid(
                column=0, row=2)
            tk.Label(frame4, text=accuracy, bg="white", font=('helvetica', 10, 'bold')).grid(column=1, row=2)
        else:
            confirm(3)
    else:
        tk.Label(frame4, text="Please load the data", bg="white", font=('helvetica', 10, 'bold')).grid(column=0, row=0)


def RF():
    for widget in frame4.winfo_children():
        widget.destroy()
    if flag1:
        if flag2:
            x = dataset[feature_cols]
            y = dataset['Heart Disease']
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
            st_x = StandardScaler()
            x_train = st_x.fit_transform(x_train)
            x_test = st_x.transform(x_test)
            global classifier
            classifier = RandomForestClassifier(criterion="entropy")
            classifier.fit(x_train, y_train)
            y_pred = classifier.predict(x_test)
            accuracy = metrics.accuracy_score(y_test, y_pred) * 100
            tk.Label(frame4, text="Accuracy of Random Forest algorithm:", bg="white",
                     font=('helvetica', 10, 'bold')).grid(column=0, row=0)
            tk.Label(frame4, text=accuracy, bg="white", font=('helvetica', 10, 'bold')).grid(column=1, row=0)

            tk.Button(frame4, text="Show tree", width=15, bg='#18F068', fg='black',
                      command=lambda: showTree("Random Forest")).place(relx=0.35, rely=0.8)
        else:
            confirm(4)
    else:
        tk.Label(frame4, text="Please load the data", bg="white", font=('helvetica', 10, 'bold')).grid(column=0, row=0)


def showTree(x):
    if x == 'Decision':
        fig = plt.figure(figsize=(40, 40))
        _ = tree.plot_tree(clf,
                           feature_names=feature_cols,
                           class_names=['0', '1'],
                           filled=True, max_depth=4)
        fig.savefig("decision_tree.png")
    else:
        fig = plt.figure(figsize=(40, 40))
        _ = tree.plot_tree(classifier.estimators_[0],
                           feature_names=feature_cols,
                           class_names=['0', '1'],
                           filled=True, max_depth=4)
        fig.savefig("random_forest_tree.png")
    plt.show()


form.mainloop()

