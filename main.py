import os
import cv2
import numpy as np
import imutils
from typing import *
from joblib import load, dump


def one_d_slice(img: np.ndarray, pos: int, row: bool) -> np.ndarray:
    return img[pos:pos + 1, :] if row else img[:, pos:pos + 1]


def find_n_black_point_on_row(ori_image, sample_point_pos, row):
    bool_threshold: int = 125

    one_d_sliced: np.ndarray = one_d_slice(ori_image, sample_point_pos, row)
    if not row:
        one_d_sliced = one_d_sliced.reshape((1, one_d_sliced.shape[0]))
    bool_arr: np.ndarray = (one_d_sliced < bool_threshold)[0]

    positions = np.where(bool_arr == 1)[0]

    out = [i for i in positions]
    popped = 0
    for index in range(1, len(positions)):
        if positions[index] - positions[index - 1] < 10:
            del out[index - popped]
            popped += 1
    return out


def get_left_side_y_values(bgr_image: np.array, sample_pos: int) -> [list, list]:
    gr_image: np.array = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    LY_begin_question_box, LY_end_question_box = find_n_black_point_on_row(gr_image, sample_pos, False)[:2]
    return LY_begin_question_box, LY_end_question_box


def get_cols_x_pos(bgr_image: np.array, y_sample_pos: int):
    gr_image: np.array = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    return find_n_black_point_on_row(gr_image, y_sample_pos, True)[:5]


def compute_rotation_angle_for_image_alignment(bgr_image: np.array, left_side_sample_pos: int) -> float:
    gr_test: np.array = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    LY_begin_question_box, LY_end_question_box = get_left_side_y_values(bgr_image, left_side_sample_pos)

    y_delta: int = 15
    LY_begin_sample_point: int = LY_begin_question_box - y_delta
    LY_end_sample_point: int = LY_end_question_box + y_delta

    top_left_x_pos_for_arctan: int = find_n_black_point_on_row(gr_test, LY_begin_sample_point, True)[0]
    bottom_left_x_pos_for_arctan: int = find_n_black_point_on_row(gr_test, LY_end_sample_point, True)[0]
    angle: float = np.arctan((top_left_x_pos_for_arctan - bottom_left_x_pos_for_arctan) /
                             (LY_end_sample_point - top_left_x_pos_for_arctan))
    return np.rad2deg(angle)


def align_black_border(bgr_image: np.array, sample_pos: int) -> np.array:
    angle: float = abs(compute_rotation_angle_for_image_alignment(bgr_image, sample_pos))
    BGR_SC_ROT_test: np.array = imutils.rotate(bgr_image, angle)
    tan_res = np.tan(np.deg2rad(angle))
    # crop for deleting black zones from rotation
    # Cropping sequence:
    #   top y crop
    #   bottom y crop
    #   left x crop
    #   right x crop
    BGR_SC_ROT_CROP_test: np.array = BGR_SC_ROT_test[round(BGR_SC_ROT_test.shape[1] * tan_res):
                                                     BGR_SC_ROT_test.shape[0] - round(
                                                         BGR_SC_ROT_test.shape[1] * tan_res),

                                                     round(BGR_SC_ROT_test.shape[0] * tan_res):
                                                     BGR_SC_ROT_test.shape[1] - round(
                                                         BGR_SC_ROT_test.shape[0] * tan_res)]
    return BGR_SC_ROT_CROP_test


def get_x_cuts(cols_x_pos, is_60_question_sim):
    correction_sequence = (2, 1, 0, 0, 0, 0, 0)
    x_cut_positions = [cols_x_pos[0 + int(not is_60_question_sim)]]
    for col in range(0 + int(not is_60_question_sim), 4 - int(not is_60_question_sim)):
        rect_width: int = (cols_x_pos[col + 1] - cols_x_pos[col]) // 7
        for cut_number in range(7):
            x_pos: int = cols_x_pos[col] + rect_width * (cut_number + 1)
            x_cut_positions.append(x_pos + correction_sequence[cut_number])
    return x_cut_positions


def raise_contrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(12, 12))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return enhanced_img


def detect_answers(bgr_image: np.array,
                   x_cut_positions: List[int], y_cut_positions: Tuple[int],
                   is_60_question_sim):
    question_multiplier: int = 15 if is_60_question_sim else 20

    letter: Tuple[str, ...] = ("L", "", "A", "B", "C", "D", "E")
    user_answer_dict: Dict[int, str] = {i: "" for i in range(1, 61 - 20 * int(not is_60_question_sim))}

    gr_image: np.array = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    # load SVM model
    load_path = os.getcwd()
    clf = load(os.path.join(load_path, "reduced.joblib"))

    for y_index in range(len(y_cut_positions) - 1):
        for x_index in range(len(x_cut_positions) - 1):

            # se sei su una colonna di numeri saltala
            if not (x_index - 1) % 7:
                continue

            x_top_left = int(not x_index % 7) * 7 + x_cut_positions[x_index]
            x_bottom_right = int(not x_index % 7) * 2 + x_cut_positions[x_index + 1]

            y_top_left: int = y_cut_positions[y_index]
            y_bottom_right: int = y_cut_positions[y_index + 1]
            crop_for_prediction: np.array = gr_image[y_top_left:y_bottom_right, x_top_left:x_bottom_right]
            crop_for_saving: np.array = cv2.resize(crop_for_prediction, (18, 18))

            category = ("QB", "QS", "QA", "CB", "CA")
            #               0     1     2     3     4

            crop_for_prediction: np.array = np.append(crop_for_saving,
                                                      [x_index % 7, int(np.mean(crop_for_prediction))])
            predicted_category_index: int = clf.predict([crop_for_prediction])[0]

            # è un quadrato di sicuro
            dump_path = os.path.join(os.getcwd(), "datasetDump")
            if x_index % 7:
                if predicted_category_index == 1:
                    save_path = ...
                    # ToDo: continuare salvando crop_for_save con il giusto nome
            cv2.imshow(category[predicted_category_index], crop_for_saving)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return None


def main(folder_name="simStorage"):
    is_60_question_sim = False
    question_multiplier: int = 15 if is_60_question_sim else 20

    folder_abs_path = os.path.join(os.getcwd(), folder_name)

    for index, img_filename in enumerate(os.listdir(folder_abs_path)):
        img = cv2.imread(os.path.join(folder_abs_path, img_filename))
        res_img: np.array = imutils.resize(img, height=900)
        x_sample_pos: int = 32
        BGR_SC_ROT_CROP_test: np.array = align_black_border(res_img, x_sample_pos)

        LY_begin_question_box, LY_end_question_box = get_left_side_y_values(BGR_SC_ROT_CROP_test, x_sample_pos)
        cols_x_pos = get_cols_x_pos(BGR_SC_ROT_CROP_test, y_sample_pos=LY_end_question_box + 20)
        x_cut_positions = get_x_cuts(cols_x_pos, is_60_question_sim)

        question_square_height: int = (LY_end_question_box - LY_begin_question_box) // question_multiplier

        y_cut_positions = tuple(
            round(y + 0.05 * y) for y in range(LY_begin_question_box - question_square_height + 6,
                                               LY_end_question_box - question_square_height, question_square_height))
        BGR_SC_ROT_CROP_test = raise_contrast(BGR_SC_ROT_CROP_test)
        user_answer_dict = detect_answers(BGR_SC_ROT_CROP_test,
                                                          x_cut_positions, y_cut_positions,
                                                          is_60_question_sim)


if __name__ == "__main__":
    main()
