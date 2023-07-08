import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import logging

logger = logging.getLogger(__file__)

LABEL = 'Label'
XLABEL = 'Counts'
YLABEL = 'Category'
OLD_GROUP_PLOT_TITLE = 'Old groups counts'
NEW_GROUP_PLOT_TITLE = 'New groups counts'

NO_TOPIC_PHRASE = '<NO_TOPIC>'
INDICATOR_COL = 'Comparison'
NEW_GROUP_VAL = 'New group'
OLD_GROUP_VAL = 'Old group'

NEW_GROUP_COLOR = '#068FFF'
OLD_GROUP_COLOR = '#A4A2AB'


def combine_words(row: pd.Series, no_topic_phrase: str) -> str:
    """
    Combine words from a row into a single string.
    Return a string with words separated by a comma.
    'outliers/no class' if '<NO_TOPIC>' in the row.
    """
    label = ', '.join(row.astype(str))
    if no_topic_phrase in label:
        label = 'outliers/no class'
    return label


def create_df_for_barh(df: pd.DataFrame,
                       indicator_col_name: str,
                       indicator_value_for_old: str,
                       indicator_value_for_new: str,
                       new_counts_col_name: str,
                       old_counts_col_name: str,
                       cols_for_label: list,
                       no_topic_phrase: str,
                       ) -> pd.DataFrame:
    """
    Create a DataFrame with only changed groups, create a label column and sets
    the type of counts columns to int.
    """
    
    df_changed = df.where(df[indicator_col_name] != indicator_value_for_old)
    df_changed = df_changed.where(
        df_changed[indicator_col_name] != indicator_value_for_new
        )
    df_changed = df_changed.dropna()
    df_changed[LABEL] = df[cols_for_label].apply(
        combine_words, args=(no_topic_phrase,), axis=1
        )
    df_changed[new_counts_col_name] = df_changed[new_counts_col_name].astype(int)
    df_changed[old_counts_col_name] = df_changed[old_counts_col_name].astype(int)
    df_changed.reset_index(inplace=True)

    output_cols = [LABEL] + \
        [new_counts_col_name, old_counts_col_name, indicator_col_name]
    return df_changed[output_cols]


def plot_changed_groups(df_changed: pd.DataFrame,
                        old_col_name: str,
                        new_col_name: str,
                        indicator_col: str,
                        filenames: list) -> plt.figure:
    """
    Plot a barh plot of changed groups.
    
    Args:
        df_changed (pd.DataFrame): The DataFrame with changed groups.
        old_col_name (str): The name of the column with old values.
        new_col_name (str): The name of the column with new values.
        indicato_col (str): The name of the column with indicator values.
        report_num (str): The number of the report to be display on the title.
        
    Returns:
        plt.figure: The figure with the plot.
    """
    df_changed = df_changed.sort_values(by=new_col_name, ascending=True)
    df_changed.reset_index(inplace=True)

    logger.info(f'Size of dataframe with changed groups: {df_changed.shape}')
    
    bar_positions = np.arange(len(df_changed))
    bar_height = 0.4
    plot_width = 8
    fig, ax = plt.subplots(figsize=(plot_width, len(df_changed) / 4 + 2))

    ax.barh(bar_positions + bar_height, df_changed[new_col_name],\
            height=bar_height, color=NEW_GROUP_COLOR, label=new_col_name)

    ax.barh(bar_positions, df_changed[old_col_name], height=bar_height,\
        color=OLD_GROUP_COLOR, label=old_col_name)

    for i, diff in enumerate(df_changed[indicator_col]):
        x = max(df_changed[new_col_name][i], df_changed[old_col_name][i]) + 1
        y = bar_positions[i] + bar_height / 2
        ax.text(x, y, f'{diff}', va='center')

    ax.legend()
    ax.set_yticks(bar_positions + bar_height)
    ax.set_yticklabels(df_changed[LABEL])

    if len(df_changed) > 46:
        y_top, y_bottom = (1.05, 1.02)
    elif len(df_changed) > 26:
        y_top, y_bottom = (1.1, 1.05)
    else:
        y_top, y_bottom = (1.2, 1.1)

    plt.text(0, y_top, s=f'Comparison Report',\
             transform=ax.transAxes, fontsize=26, verticalalignment='top',\
                ha='left')
    plt.text(0, y_bottom, f'{filenames[0]} vs {filenames[1]}',\
            transform=ax.transAxes, fontsize=12, verticalalignment='top',\
                ha='left')
    # plt.title('Comparison Report', fontsize=32, pad=20, loc='left', )
    # plt.subtitle('The groups that have changed', fontsize=16, pad=20, loc='left')
    plt.xlabel(XLABEL)
    plt.ylabel(YLABEL)
    plt.tight_layout()
    plt.subplots_adjust(right=2)
    
    return fig


def create_single_bar_barh(df,
                           plot_title: str,
                           vals_col_name: str,
                           label_col_name: str,
                           new_group=True) -> plt.figure:
    """"
    Create a barh plot of a single group.
    """
    df = df.sort_values(by=vals_col_name, ascending=True)
    df.reset_index(inplace=True)

    

    figsize = (8, len(df) / 5 + 1) if len(df) > 10 else (8, 6)
    fig, ax = plt.subplots(figsize=figsize)

    bar_positions = np.arange(len(df))
    color = NEW_GROUP_COLOR if new_group else OLD_GROUP_COLOR
    plt.barh(df[label_col_name], df[vals_col_name], color=color)

    for i, diff in enumerate(df[vals_col_name]):
        x = df[vals_col_name][i] + 1
        y = bar_positions[i]
        ax.text(x, y, f'{diff}', va='center')

    plt.xlabel(XLABEL)
    plt.ylabel(YLABEL)
    plt.title(f'{plot_title}')
    plt.tight_layout()
    plt.subplots_adjust(right=2)

    return fig

def create_text(text: str) -> plt.figure:
    """
    Create a figure with a text.
    """
    fig = plt.figure(visible=True, figsize=(8, 1))
    ax = fig.add_subplot(111)
    ax.text(0.5, 0.5, text, ha='right', va='center', fontsize=12)
    plt.axis('off')
    return fig


def plot_group(comp_report_df: pd.DataFrame,
               vals_col_name: str,
               indicator_value: str,
               columns_for_label: list,
               indicator_col_name: str,
               plot_title: str,
               no_topic_phrase: str,
               new_group: bool
               ) -> plt.figure or None:
    """
    Create a barh plot of a single group (no changes).
    Color of the bar is determined by the new_group parameter.

    Args:
        comp_report_df (pd.DataFrame): The DataFrame with the comparison report.
        vals_col_name (str): The name of the column with values.
        indicator_value (str): The value of the indicator column that tells if 
            the group is new, old or simply has changed quantity.
        columns_for_label (list): The list of columns to be used for the label.
        indicator_col_name (str): The name of the indicator column.
        plot_title (str): The title of the plot.
        no_topic_phrase (str): The phrase to be used if there are no topics.
        new_group (bool): If True, the color of the bar is NEW_GROUP_COLOR,
            otherwise OLD_GROUP_COLOR.

    Returns:
        plt.figure: The bar horizontal plot.
    """

    df = comp_report_df[columns_for_label + [vals_col_name, indicator_col_name]].copy()    
    df.where(df[indicator_col_name] == indicator_value, inplace=True)
    df.dropna(inplace=True)
    if len(df) == 0:
        fig = create_text(f'There are no "{indicator_value}" groups.')
    else:
        df[LABEL] = df[columns_for_label].apply(
            lambda x: combine_words(x, no_topic_phrase), axis=1
            )
        
        df[vals_col_name] = pd.to_numeric(df[vals_col_name], errors='coerce')
        df.dropna(inplace=True)
        
        df[vals_col_name] = df[vals_col_name].astype(int)

        df.reset_index(inplace=True)

        df = df[[LABEL] + [vals_col_name]]
    
        fig = create_single_bar_barh(df,
                                    vals_col_name=vals_col_name,
                                    label_col_name=LABEL,
                                    plot_title=plot_title,
                                    new_group=new_group
                                    )
    
    return fig


def create_table(df: pd.DataFrame) -> plt.figure:
    """
    Converts a DataFrame to a  matplotlib figure (table).
    """
    fig, ax = plt.subplots()

    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(12)

    num_rows, num_cols = df.shape

    table.scale(3.5, num_rows / (num_cols + 1))
    ax.axis('off')

    return fig


def save_to_pdf(figures: list, output_file_path: str) -> None:
    """
    Save a list of figures to a pdf file.
    
    Args:
        figures (list): A list of figures.
        output_file_name (str): The name of the output file.
    """

    with PdfPages(output_file_path) as pdf:
        for fig in figures:
            if fig is not None:
                pdf.savefig(fig, bbox_inches='tight')
        plt.close()


def create_pdf_comaprison_report(
        df: pd.DataFrame,
        old_col_name: str,
        new_col_name: str,
        output_file_path: str,
        cols_for_label: list,
        cols_for_old_label: list,
        filenames: list,
        old_col_value: str=OLD_GROUP_VAL,
        new_col_value: str=NEW_GROUP_VAL,
        indicator_col_name: str=INDICATOR_COL,
        no_topic_phrase: str=NO_TOPIC_PHRASE,
        new_group_plot_title: str=NEW_GROUP_PLOT_TITLE,
        old_group_plot_title: str=OLD_GROUP_PLOT_TITLE,
        ) -> list:
    """
    Calls all plot generation functions and saves the plots to a pdf file.
    """
    df_changed = create_df_for_barh(df,
                                    indicator_col_name=indicator_col_name,
                                    indicator_value_for_new=new_col_value,
                                    indicator_value_for_old=old_col_value,
                                    new_counts_col_name=old_col_name,
                                    old_counts_col_name=new_col_name,
                                    cols_for_label=cols_for_label,
                                    no_topic_phrase=no_topic_phrase
                                    )
    
    fig_changed_groups = plot_changed_groups(df_changed,
                                             old_col_name,
                                             new_col_name,
                                             indicator_col_name,
                                             filenames
                                             )    
    
    fig_new_group = plot_group(df,
                               vals_col_name=new_col_name,
                               indicator_value=new_col_value,
                               columns_for_label=cols_for_label,
                               indicator_col_name=indicator_col_name,
                               plot_title=new_group_plot_title,
                               no_topic_phrase=no_topic_phrase,
                               new_group=True)
    
    fig_old_group = plot_group(df,
                               vals_col_name=old_col_name,
                               indicator_value=old_col_value,
                               columns_for_label=cols_for_old_label,
                               indicator_col_name=indicator_col_name,
                               plot_title=old_group_plot_title,
                               no_topic_phrase=no_topic_phrase,
                               new_group=False)

    table = create_table(df)

    logger.info('Creating comparison report PDF...')

    fig_list = [fig_changed_groups, fig_new_group, fig_old_group, table]

    save_to_pdf(fig_list,
                output_file_path)
    
    logger.info(f'Comparison report PDF was saved to {output_file_path}')
