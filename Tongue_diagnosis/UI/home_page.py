# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'home_page.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_HomePageWindow(object):
    def setupUi(self, HomePageWindow):
        HomePageWindow.setObjectName("HomePageWindow")
        HomePageWindow.resize(1661, 910)
        HomePageWindow.setBaseSize(QtCore.QSize(2, 0))
        HomePageWindow.setStyleSheet("QMainWindow#HomePageWindow{background-image:url(D:/graduation_design/Tongue_diagnosis/UI/img/320.jpg);}")
        self.centralwidget = QtWidgets.QWidget(HomePageWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(1380, 790, 181, 51))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.pushButton.setFont(font)
        self.pushButton.setStyleSheet("QPushButton {\n"
"    border: 2px solid #8f8f91;\n"
"    border-radius: 6px;\n"
"    background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,\n"
"                                      stop: 0 #f6f7fa, stop: 1 #dadbde);\n"
"    min-width: 80px;\n"
"}\n"
"\n"
"QPushButton:hover\n"
"{\n"
"    background-color:qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,\n"
"                                      stop: 0 #dadbde, stop: 1 #f6f7fa);\n"
"}\n"
"\n"
"\n"
"QPushButton:pressed {\n"
"    background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,\n"
"                                      stop: 0 #dadbde, stop: 1 #f6f7fa);\n"
"}\n"
"\n"
"QPushButton:flat {\n"
"    border: none; /* no border for a flat push button */\n"
"}\n"
"\n"
"QPushButton:default {\n"
"    border-color: navy; /* make the default button prominent */\n"
"}")
        self.pushButton.setObjectName("pushButton")
        HomePageWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(HomePageWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1661, 22))
        self.menubar.setObjectName("menubar")
        HomePageWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(HomePageWindow)
        self.statusbar.setObjectName("statusbar")
        HomePageWindow.setStatusBar(self.statusbar)

        self.retranslateUi(HomePageWindow)
        QtCore.QMetaObject.connectSlotsByName(HomePageWindow)

    def retranslateUi(self, HomePageWindow):
        _translate = QtCore.QCoreApplication.translate
        HomePageWindow.setWindowTitle(_translate("HomePageWindow", "欢迎使用舌苔检测系统！"))
        self.pushButton.setText(_translate("HomePageWindow", "进入系统"))
