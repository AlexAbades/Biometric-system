import cv2
import numpy as np


def plot_embeding_grid(emb_dist, input_size):
    """
    ATRIBUTES
    ---------
    
    emb_dist -> square matrix of embeding distances. Must be same len as the images 
    input_size, size of the image 
    """

    n = len(emb_dist)
    rows = []
    for i in range(n):
        row = []
        for j in range(n):
            # create small colorful image from value in distance matrix
            value = emb_dist[i][j]
            cell = np.empty(input_size)
            cell.fill(value)
            cell = (cell * 255).astype(np.uint8)
            # color depends on value: blue is closer to 0, green is closer to 1
            img = cv2.applyColorMap(cell, cv2.COLORMAP_WINTER)

            # add distance value as text centered on image
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f"{value:.4f}"
            textsize = cv2.getTextSize(text, font, 1, 2)[0]
            text_x = (img.shape[1] - textsize[0]) // 2
            text_y = (img.shape[0] + textsize[1]) // 2
            cv2.putText(
                img, text, (text_x, text_y), font, 2, (255, 255, 255), 2, cv2.LINE_AA,
            )
            row.append(img)
        rows.append(np.concatenate(row, axis=1))
    grid = np.concatenate(rows)
    return grid



def similarity_image(images, cos_similarity, tag):
    """
    ATRIBUTES
    ---------
    images -> List of images 
    cos_similarity -> matrix with values of similarity  
    """
    if len(images) != len(cos_similarity):
        print('Must provide same number of images as dimensions of square matrix')

    input_size = images[0].shape[:-1]
    similarity_grid = plot_embeding_grid(cos_similarity, input_size)
    # pad similarity grid with images of faces
    horizontal_grid = np.hstack(images)
    vertical_grid = np.vstack(images)
    zeros = np.zeros((*input_size, 3))
    vertical_grid = np.vstack((zeros, vertical_grid))
    result = np.vstack((horizontal_grid, similarity_grid))
    result = np.hstack((vertical_grid, result))


    cv2.imwrite(f'{tag}.jpg', result)