from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys
import effcient_Net.tongue_crack.crack_predict as crack
import effcient_Net.tongue_coated.coated_predict as coated
import effcient_Net.tongue_color.color_predict as color
import effcient_Net.tongue_indentation.indent_predict as indent
import os
from UI.home_page import Ui_HomePageWindow
from UI.seg_tongue_work import Ui_MainWindow
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
class mainWin(QMainWindow, Ui_HomePageWindow):
    def __init__(self,parent=None):
        super(mainWin, self).__init__(parent)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.login)
    def login(self):
        main_win.close()
        second_main.show()

class secondmain(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(secondmain,self).__init__(parent)
        self.setupUi(self)
        self.loadtongue.clicked.connect(self.openimage)
        self.loadtongue.setFlat(True)
        self.start.clicked.connect(self.start_predict)
        self.back_btn.clicked.connect(self.back_home)
    def openimage(self):
        self.tongue_color.setText("")
        self.tai_color.setText("")
        self.tongue_shape2.setText("")
        self.tongue_shape3.setText("")
        self.sugtext.setText("")
        imgName,imgType = QFileDialog.getOpenFileName(self,"打开图片", "../data/test", "*.jpg;*.tif;*.png;;All Files(*)")
        if imgName == "":
            return 0
        jpg = QPixmap(imgName).scaled(self.uesr_tongue.width(),self.uesr_tongue.height())
        jpg.save('./user_load.jpg')
        self.sugtext.setText("")
        self.uesr_tongue.setPixmap(jpg)
    def back_home(self):
        self.uesr_tongue.setPixmap(QPixmap('img/back2.jpg'))
        self.tongue_recognition.setPixmap(QPixmap('img/back2.jpg'))
        self.tongue_color.setText("")
        self.tai_color.setText("")
        self.tongue_shape2.setText("")
        self.tongue_shape3.setText("")
        self.sugtext.setText("")
        second_main.close()
        main_win.show()
    def start_predict(self):
        self.tongue_color.setText("")
        self.tai_color.setText("")
        self.tongue_shape2.setText("")
        self.tongue_shape3.setText("")
        self.tongue_recognition.setPixmap(
        QPixmap('./img/1.png').scaled(self.tongue_recognition.width(), self.tongue_recognition.height()))
        crack_class, crack_prob = crack.main('./user_load.jpg')
        coated_class, coated_prob = coated.main('./user_load.jpg')
        color_class, color_prob = color.main('./user_load.jpg')
        indent_class, indent_prob = indent.main('./user_load.jpg')
        if crack_class == 'crack':
            crack_res = "裂纹舌"
        else:
            crack_res = "无裂纹"

        if indent_class == 'normal':
            indent_res = "无齿痕"
        else:
            indent_res = "齿痕舌"

        if coated_class == "white":
            coated_res = "白苔"
        else:
            if coated_class == "yellow":
                coated_res = "黄苔"
            else:
                coated_res = "无苔"
        if color_class == "red":
            color_res = "淡红舌"
        else:
            if color_class == "white":
                color_res = "淡白舌"
            else:
                color_res = "深红舌"

        self.tongue_shape2.setText("{}".format(crack_res))
        self.tongue_shape3.setText("{}".format(indent_res))
        self.tongue_color.setText("{}".format(color_res))
        self.tai_color.setText("{}".format(coated_res))

        if color_class == 'white' and crack_class == 'crack' and indent_class == 'indentation' and coated_class == "white":
            self.sugtext.setText("舌淡白，白舌苔，有齿痕，有裂纹:\n燥热伤津，阴液亏损，脾虚湿侵，脾失健运，湿邪内侵，精微不能濡养舌体。")
        elif color_class == 'white' and crack_class == 'crack' and indent_class == 'indentation' and coated_class == "yellow":
            self.sugtext.setText("舌淡白，黄舌苔，有齿痕，有裂纹:\n风热表证,或风寒化热入里，热势轻浅,脾虚湿侵，脾失健运，湿邪内侵，精微不能濡养舌体。")
        elif color_class == 'white' and crack_class == 'crack' and indent_class == 'indentation' and coated_class == "nocoated":
            self.sugtext.setText("舌淡白，有齿痕，有裂纹:\n热势轻浅，脾虚湿侵，脾失健运，湿邪内侵，精微不能濡养舌体。")

        elif color_class == 'white' and crack_class == 'crack' and indent_class == 'normal' and coated_class == "white":
            self.sugtext.setText("舌淡白，白舌苔，有裂纹:\n燥热伤津，阴液亏损,血虚不润,血虚不能上荣于活,精微不能濡养舌体。")
        elif color_class == 'white' and crack_class == 'crack' and indent_class == 'normal' and coated_class == "yellow":
            self.sugtext.setText("舌淡白，黄舌苔，有裂纹:\n血虚不润,血虚不能上荣于活,精微不能濡养舌体，风热表证,或风寒化热入里，热势轻浅。")
        elif color_class == 'white' and crack_class == 'crack' and indent_class == 'normal' and coated_class == "nocoated":
            self.sugtext.setText("舌淡白，有裂纹:\n血虚不润,血虚不能上荣于活,精微不能濡养舌体。")

        elif color_class == 'white' and crack_class == 'normal' and indent_class == 'indentation' and coated_class == "white":
            self.sugtext.setText("舌淡白，白舌苔，有齿痕：\n表证、寒证，主脾虚、血虚，水湿内盛证，舌胖大而多齿痕多属脾虚或湿困")
        elif color_class == 'white' and crack_class == 'normal' and indent_class == 'indentation' and coated_class == "yellow":
            self.sugtext.setText("舌淡白，黄舌苔，有齿痕：\n里证，热证主脾虚、血虚，水湿内盛证，舌胖大而多齿痕多属脾虚或湿困")
        elif color_class == 'white' and crack_class == 'normal' and indent_class == 'indentation' and coated_class == "nocoated":
            self.sugtext.setText("舌淡白，有齿痕：\n主脾虚、血虚，水湿内盛证，舌胖大而多齿痕多属脾虚或湿困")

        elif color_class == 'white' and crack_class == 'normal' and indent_class == 'normal' and coated_class == "white":
            self.sugtext.setText("舌淡白，白舌苔：\n血虚，也主表证、寒证")
        elif color_class == 'white' and crack_class == 'normal' and indent_class == 'normal' and coated_class == "yellow":
            self.sugtext.setText("舌淡白，黄舌苔：\n血虚，主里证，热证")
        elif color_class == 'white' and crack_class == 'normal' and indent_class == 'normal' and coated_class == "nocoated":
            self.sugtext.setText("舌淡白：\n血虚")

        elif color_class == 'red' and crack_class == 'normal' and indent_class == 'normal' and coated_class == 'nocoated':
            self.sugtext.setText("舌淡红，无舌苔：\n虚热证")
        elif color_class == 'red' and crack_class == 'normal' and indent_class == 'normal' and coated_class == 'white':
            self.sugtext.setText("舌淡红，白舌苔：\n心气充足，胃气旺盛，气血调和，常见于正常人或病情轻浅阶段")
        elif color_class == 'red' and crack_class == 'normal' and indent_class == 'normal' and coated_class == 'yellow':
            self.sugtext.setText("舌淡红，黄舌苔：虚热证，主里证")

        elif color_class == 'red' and crack_class == 'crack' and indent_class == 'normal' and coated_class == 'nocoated':
            self.sugtext.setText("舌淡红，无舌苔，有裂纹：\n虚热证，精血亏虚或阴津耗损，舌体失养，血虚之候，可能为全身营养不良")
        elif color_class == 'red' and crack_class == 'crack' and indent_class == 'normal' and coated_class == 'white':
            self.sugtext.setText("舌淡红，白舌苔，有裂纹：\n虚热证，主表证，精血亏虚或阴津耗损，舌体失养，血虚之候")
        elif color_class == 'red' and crack_class == 'crack' and indent_class == 'normal' and coated_class == 'yellow':
            self.sugtext.setText("舌淡红，黄舌苔，有裂纹：\n虚热证，风寒化热入里，热势轻浅，精血亏虚或阴津耗损，舌体失养，血虚之候")

        elif color_class == 'red' and crack_class == 'normal' and indent_class == 'indentation' and coated_class == 'yellow':
            self.sugtext.setText("舌淡红，黄舌苔，有齿痕:\n气虚证或脾虚证，气血不足，风寒化热入里，热势轻浅。")
        elif color_class == 'red' and crack_class == 'normal' and indent_class == 'indentation':
            self.sugtext.setText("舌淡红，有齿痕:\n气虚证或脾虚证，气血不足。")

        elif color_class == 'red' and crack_class == 'crack' and indent_class == 'indentation' and coated_class == 'yellow':
            self.sugtext.setText("舌淡红，黄舌苔，有裂纹，有齿痕:\n气虚证或虚热证，风寒化热入里，热势轻浅，精血亏虚或阴津耗损，舌体失养，气血不足。")
        elif color_class == 'red' and crack_class == 'crack' and indent_class == 'indentation':
            self.sugtext.setText("舌淡红，有裂纹，有齿痕:\n气虚证或虚热证，精血亏虚或阴津耗损，舌体失养，气血不足。")

        elif color_class == 'crimson' and crack_class == 'crack' and coated_class == 'white':
            self.sugtext.setText("舌深红，白舌苔，有裂纹:\n热症，热盛伤津，邪热内盛,阴液大伤，或阴虚液损，使舌体失于濡润,舌面萎缩。")
        elif color_class == 'crimson' and crack_class == 'crack' and coated_class == 'yellow':
            self.sugtext.setText("舌深红，黄舌苔，有裂纹:\n热症，热盛伤津，风寒化热入里，邪热内盛，阴液大伤。或阴虚液损，使舌体失于濡润，舌面萎缩，舌体失养。")
        elif color_class == 'crimson'and crack_class == 'crack' and coated_class == 'nocoated':
            self.sugtext.setText("舌深红，无舌苔，有裂纹:\n热症，热盛伤津，邪热内盛，阴液大伤，或阴虚液损，使舌体失于濡润，舌面萎缩，阴虚火旺。或热病后期阴液耗损。")

        elif color_class == 'crimson' and crack_class == 'normal' and indent_class == 'indentation' and coated_class == 'nocoated':
            self.sugtext.setText("舌深红，无舌苔，有齿痕:\n热症，久病阴虚火旺,或热病后期阴液耗损，水湿内盛证，舌胖大而多齿痕多属脾虚或湿困。")
        elif color_class == 'crimson' and crack_class == 'normal' and indent_class == 'normal' and coated_class == 'nocoated':
            self.sugtext.setText("舌深红，无舌苔:\n热症，久病阴虚火旺,或热病后期阴液耗损。")
        elif color_class == 'crimson' and crack_class == 'normal' and indent_class == 'normal' and coated_class == 'yellow':
            self.sugtext.setText("舌深红,黄苔:\n热症，温热病热入营血，或脏腑内热炽盛,风热表证,或风寒化热入里，热势轻浅。")
        elif color_class == 'crimson' and crack_class == 'normal' and indent_class == 'indentation' and coated_class == 'white':
            self.sugtext.setText("舌深红，白舌苔，有齿痕:\n热症，久病阴虚火旺,或热病后期阴液耗损，水湿内盛证，舌胖大而多齿痕多属脾虚或湿困。")
        elif color_class == 'crimson'and crack_class == 'normal' and indent_class == 'normal' and coated_class == 'white':
            self.sugtext.setText("舌深红，白苔:\n热症，温热病热入营血，或脏腑内热炽盛。")
        elif color_class == 'crimson'and crack_class == 'normal' and indent_class == 'indentation' and coated_class == 'white':
            self.sugtext.setText("舌深红，白舌苔，有齿痕:\n热症，温热病热入营血，或脏腑内热炽盛，水湿内盛证，舌胖大而多齿痕多属脾虚或湿困。")





if __name__ == '__main__':
    app = QApplication(sys.argv)
    #初始化窗口
    main_win = mainWin()
    second_main = secondmain()
    #second_main.setFixedSize(1666,920)
    main_win.setFixedSize(1666,870)
    #main_win.move((QApplication.desktop().width()-main_win.width())/2,(QApplication.desktop().height()-main_win.height())/13)

    main_win.show()
    sys.exit(app.exec_())