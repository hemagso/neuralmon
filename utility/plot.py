import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import skimage.color as color
import numpy as np
import os

from sklearn.metrics import recall_score, precision_score, accuracy_score

from .load import load_type_dict

def plot_all(pkm):
    main_folder = "./sprites/pokemon/main-sprites/"
    img_folders = sorted([f for f in os.listdir(main_folder) if not f.startswith('.')])
    file = "{n}.png".format(n=pkm)    
    plt.figure(figsize=(14,4))
    for idx, folder in enumerate(img_folders):            
        img = mpimg.imread(os.path.join(main_folder,folder,file))
        plt.subplot(2,7,idx+1)
        plt.imshow(img)
    plt.show()
    
def plot_chain(game_folder,pkmns):
    main_folder = "./sprites/pokemon/main-sprites/"
    img_folder = os.path.join(main_folder,game_folder)
    n = len(pkmns)
    plt.figure(figsize=(8,24))
    for idx, pkm in enumerate(pkmns):
        file = "{n}.png".format(n=pkm)
        img = mpimg.imread(os.path.join(img_folder,file))
        plt.subplot(1,n,idx+1)
        plt.imshow(img)
    plt.show()        
    
def plot_sprite(sprite,type_1=1,type_2=None,pred=None,type_dict = load_type_dict(),save=None,save_path="./classification"):
    #Definindo as dimensões do Grid
    if pred:
        grid_rows = 2
        grid_cols = 2
        figsize = (8, 4.4)        
        width_ratios = (1, 1)
        sprite_grid = 0
        pred_grid = 1
        type_grid = 2
    else:
        grid_rows = 2
        grid_cols = 1
        figsize = (4, 4.4)
        width_ratios = (1,)
        sprite_grid = 0
        type_grid = 1
        
        
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(grid_rows,grid_cols, height_ratios = (10,1), width_ratios = width_ratios)
    
    #Plotando o sprite do Pokemon
    ax_sprite = plt.subplot(gs[sprite_grid])
    ax_sprite.imshow(color.hsv2rgb(sprite))    
    
    #Plotando o tipo verdadeiro do pokemon
    ax_type = plt.subplot(gs[type_grid])
    plt.axis("off")
    
    type_box_01 = patches.Rectangle(
        (0,0),
        0.5 if type_2 else 1,  
        1,  
        fc = type_dict[type_1]["color"],
        ec = "#FFFFFF"
    )    
    ax_type.add_patch(type_box_01)
    ax_type.annotate(type_dict[type_1]["label"], (0.25 if type_2 else 0.5, 0.5), color='w', weight='bold', 
                fontsize=12, ha='center', va='center')    
    if type_2:
        type_box_02 = patches.Rectangle(
            (0.5,0),
            0.5,  
            1,  
            fc = type_dict[type_2]["color"],
            ec = "#FFFFFF"
        )            
        ax_type.add_patch(type_box_02)
        ax_type.annotate(type_dict[type_2]["label"], (0.75, 0.5), color='w', weight='bold', 
                    fontsize=12, ha='center', va='center')       

    
    #Plotando as previsões
    if pred:
        ax_pred = plt.subplot(gs[pred_grid])
        plt.axis("off")

        pred_list = list(pred.items())
        pred_list = sorted(pred_list, key = lambda x: x[1],reverse = True)    
        for idx, (pred_type, pred_prob) in enumerate(pred_list):
            pred_box = patches.Rectangle(
                (0,0.8-0.2*idx),
                0.5,
                0.2,
                fc = type_dict[pred_type]["color"],
                ec = "#FFFFFF"            
            )
            ax_pred.add_patch(pred_box)
            ax_pred.annotate(type_dict[pred_type]["label"], (0.25, 0.9-0.2*idx), color='#FFFFFF', weight='bold', 
                        fontsize=12, ha='center', va='center')       
            ax_pred.annotate("{:.0%}".format(pred_prob), (0.75, 0.9-0.2*idx), color='#000000', weight='bold', 
                        fontsize=16, ha='center', va='center')   
    if save:
        correct_path = os.path.join(save_path,"correct")
        wrong_path = os.path.join(save_path,"wrong")
        if not os.path.exists(correct_path):
            os.makedirs(correct_path)
        if not os.path.exists(wrong_path):
            os.makedirs(wrong_path)      
        if type_1 == pred_list[0][0]:
            save_file = os.path.join(correct_path,save)            
        else:
            save_file = os.path.join(wrong_path,save)
        fig.savefig(save_file)                                   

def plot_record(rec):
    plot_sprite(
        rec["sprite"],
        type_1 = rec["type_01"],
        type_2 = None if np.isnan(rec["type_02"]) else rec["type_02"]
    )    
    
def plot_evaluation(label,y_true,y_pred,type_dict=load_type_dict()):
    y_pred = np.argmax(y_pred,axis=1)+1
    y_true = np.argmax(y_true,axis=1)+1    
    #Evaluate model metrics over input data
    recall = recall_score(y_true, y_pred, average=None)
    precision = precision_score(y_true, y_pred, average=None)
    accuracy = accuracy_score(y_true, y_pred)
    
    #Create grid for plotting
    fig = plt.figure(figsize=(8.8,5))
    gs = gridspec.GridSpec(2,1, height_ratios = (1,9))
       
    #Plotting model-level metrics
    ax = plt.subplot(gs[0])     
    ax.axis("off")    
    ax.annotate("{} Accuracy = {:.0%}".format(label,accuracy), (0.5, 0.5), color='#000000', 
                fontsize=18, ha='center', va='center')     
        
    #Ploting class-level metrics
    ax = plt.subplot(gs[1]) 
    ax.axis("off")    
    
    #In some cases, there are no records of some classes (usually 3:Flying) Here, we fill
    #up the missing classes with 'None' values.
    unique_labels = np.unique(np.vstack([y_true,y_pred]))
    metrics = dict( (key, {"recall" : None, "precision" : None}) for key in range(1,19))
    for key, v_recall, v_precision in zip(unique_labels, recall, precision):
        metrics[key]["recall"] = v_recall
        metrics[key]["precision"] = v_precision

    #Writing the headers of the class table
    ax.annotate("Precision", (0.27, 19/20), color='#000000', weight='bold', 
                fontsize=12, ha='center', va='center')     
    ax.annotate("Recall", (0.4, 19/20), color='#000000', weight='bold', 
                fontsize=12, ha='center', va='center')        
    ax.annotate("Precision", (0.77, 19/20), color='#000000', weight='bold', 
                fontsize=12, ha='center', va='center')        
    ax.annotate("Recall", (0.9, 19/20), color='#000000', weight='bold', 
                fontsize=12, ha='center', va='center')            
    
    #Writing the metrics for each class
    for i, (pkm_type, metric) in enumerate(metrics.items()):
        column = int(i/9)
        row = i % 9 + 1
        left = column*0.5
        top = 1.0-(row+1)*1/10
        type_box = patches.Rectangle(
            (left,top),
            0.2,
            1/9,
            fc = type_dict[pkm_type]["color"],
            ec = "#FFFFFF"            
        ) 
        ax.add_patch(type_box)
        ax.annotate(type_dict[pkm_type]["label"], (left+0.1, top+1/20), color='#FFFFFF', weight='bold', 
                    fontsize=12, ha='center', va='center')  
        #Precision
        if metric["precision"] is not None:
            ax.annotate("{:.0%}".format(metric["precision"]), (left+0.27, top+1/20), color='#000000', 
                        fontsize=14, ha='center', va='center')    
        #Recall
        if metric["recall"] is not None:
            ax.annotate("{:.0%}".format(metric["recall"]), (left+0.40, top+1/20), color='#000000', 
                        fontsize=14, ha='center', va='center')          