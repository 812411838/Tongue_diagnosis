import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties
from effcient_Net.tongue_coated.coated_predict import main as coater_predict
from effcient_Net.tongue_color.color_predict import main as color_predict
from effcient_Net.tongue_crack.crack_predict import main as crack_predict
from effcient_Net.tongue_indentation.indent_predict import main as indent_predict
if __name__ == '__main__':
    img_path = "../data/test/19.png"
    font = FontProperties(fname='../msyhbd.ttc')

    coated_class, coated_prob = coater_predict(img_path)
    color_class, color_prob = color_predict(img_path)
    crack_class, crack_prob = crack_predict(img_path)
    indent_class, indent_prob = indent_predict(img_path)


    # coated_res = "苔色 : {} prob: {:.3}".format(coated_class,coated_prob)
    # color_res = "舌色 : {}  prob: {:.3}".format(color_class, color_prob)
    # crack_res = "裂纹 : {}  prob: {:.3}".format(crack_class, crack_prob)
    # indent_res = "齿痕 : {}  prob: {:.3}".format(indent_class, indent_prob)

    coated_res = "苔色 : {}".format(coated_class)
    color_res = "舌色 : {} ".format(color_class)
    crack_res = "裂纹 : {} ".format(crack_class)
    indent_res = "齿痕 : {}  ".format(indent_class)


    res = "{}   {}\n{}   {}\n".format(coated_res, color_res, crack_res, indent_res)
    plt.title(res,fontproperties=font)
    plt.show()


























