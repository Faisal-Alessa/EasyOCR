import easyocr.easyocr as easyocr
import cv2
import pandas as pd

def save_preds(img, img_path, prepared_path, preds):
    # init variables
    restuls_csv = dict()
    # extract preds, cords, and text
    for idx, pred in enumerate(preds):
        x1, y1 = pred[0][0]
        x2, y2 = pred[0][2]
        box = img[max(0, int(y1)):max(0, int(y2)), max(0, int(x1)):max(0, int(x2))] # make sure cords are positive ints
        text = pred[1]

        img_name = img_path.split('/')[-1]
        box_name = f'{idx}_{img_name}'

        try:
            # save img
            cv2.imwrite(f'{prepared_path}/{box_name}', box)
            # append labels in a csv file
            restuls_csv['filename'] = [box_name]
            restuls_csv['words'] = [text]
        except Exception as e:
            print('Empty box, can\'t save the image, Exception: ', e)
    
    results_df = pd.DataFrame(restuls_csv)
    results_df.to_csv('preds.csv', mode='a', index=False, header=False)


def native_pipeline(prepared_path, img_path):
    img = cv2.imread(img_path)
    assert img is not None, f"Can't read or empty image. File:{img_path}"
    reader = easyocr.Reader(lang_list=['en', 'ar'], verbose=False)
    result = reader.readtext(img)
    save_preds(img, img_path, prepared_path, result)

