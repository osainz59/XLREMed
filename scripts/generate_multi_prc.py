import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
import pandas as pd
import seaborn as sns

general_relations = [1,2,3,4,5,6]
contextual_relations = [7, 8, 9]
action_roles = [10, 11]
predicate_roles = [12, 13]

def main():
    
    models = [
        ('XLMem', '-',       
            'evaluation/ehealthkd/XLM-17_gold/dev-XLM-17_gold-y_true.npy', 
            'evaluation/ehealthkd/XLM-17_gold/dev-XLM-17_gold-y_probs.npy'),
        ('XLMem*', '*',      
            'evaluation/ehealthkd/XLM-17_gold+silver/dev-XLM-17_gold+silver-y_true.npy', 
            'evaluation/ehealthkd/XLM-17_gold+silver/dev-XLM-17_gold+silver-y_probs.npy'),
        ('XLMem*+MTB', '--',  
            'evaluation/ehealthkd/XLM-17-MTB_gold+silver/dev-EHealthKD-y_true.npy', 
            'evaluation/ehealthkd/XLM-17-MTB_gold+silver/dev-EHealthKD-y_probs.npy'),
    ]

    fig = plt.figure(figsize=(10, 10))
    plt.title(f"Precision/Recall curves")
    # Plot the F-Scores
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    global_df = pd.DataFrame()
    for name, marker, y_true, y_probs in models:
        y_probs = np.load(y_probs)
        y_true = label_binarize(np.load(y_true), list(range(y_probs.shape[1])))

        

        pre, rec, _ = precision_recall_curve(y_true[:, 1:].ravel(), y_probs[:, 1:].ravel())
        #plt.plot(rec, pre, f'b{marker}', lw=2)
        df_average = pd.DataFrame()
        df_average['pre'] = pre
        df_average['rec'] = rec
        df_average['Relations'] = 'micro-average'

        pre, rec, _ = precision_recall_curve(y_true[:, general_relations].ravel(), 
                                             y_probs[:, general_relations].ravel())
        #plt.plot(rec, pre, f'g{marker}', lw=1)
        df_general = pd.DataFrame()
        df_general['pre'] = pre
        df_general['rec'] = rec
        df_general['Relations'] = 'general'

        pre, rec, _ = precision_recall_curve(y_true[:, contextual_relations].ravel(), 
                                             y_probs[:, contextual_relations].ravel())
        #plt.plot(rec, pre, f'r{marker}', lw=1)
        df_contextual = pd.DataFrame()
        df_contextual['pre'] = pre
        df_contextual['rec'] = rec
        df_contextual['Relations'] = 'contextual'

        pre, rec, _ = precision_recall_curve(y_true[:, action_roles].ravel(), 
                                             y_probs[:, action_roles].ravel())
        #plt.plot(rec, pre, f'm{marker}', lw=1)
        df_action = pd.DataFrame()
        df_action['pre'] = pre
        df_action['rec'] = rec
        df_action['Relations'] = 'action-roles'

        pre, rec, _ = precision_recall_curve(y_true[:, predicate_roles].ravel(), 
                                             y_probs[:, predicate_roles].ravel())
        #plt.plot(rec, pre, f'c{marker}', lw=1)
        df_predicate = pd.DataFrame()
        df_predicate['pre'] = pre
        df_predicate['rec'] = rec
        df_predicate['Relations'] = 'predicate-roles'

        df = pd.concat([df_average, df_general, df_contextual, df_action, df_predicate])
        df['Model name'] = name

        global_df = pd.concat([global_df, df])


    ax = sns.lineplot(x="rec", y="pre", err_style=None,
                      hue="Relations", style="Model name", data=global_df)

    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    fig.savefig('evaluation/ehealthkd/multi-pr_curve.png')

if __name__ == "__main__":
    main()