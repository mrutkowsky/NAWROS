import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import logging

logger = logging.getLogger(__file__)

XLABEL = 'Counts'
YLABEL = 'Category'

NEW_GROUP_COLOR = '#69DB24'
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
    df_changed['Label'] = df[cols_for_label].apply(
        combine_words, args=(no_topic_phrase,), axis=1
        )
    df_changed[new_counts_col_name] = df_changed[new_counts_col_name].astype(int)
    df_changed[old_counts_col_name] = df_changed[old_counts_col_name].astype(int)
    df_changed.reset_index(inplace=True)

    output_cols = ['Label'] + \
        [new_counts_col_name, old_counts_col_name, indicator_col_name]
    return df_changed[output_cols]


def plot_changed_groups(df_changed: pd.DataFrame,
                        old_col_name: str,
                        new_col_name: str,
                        indicato_col: str) -> plt.figure:
    """
    Plot a barh plot of changed groups.
    
    Args:
        df_changed (pd.DataFrame): The DataFrame with changed groups.
        old_col_name (str): The name of the column with old values.
        new_col_name (str): The name of the column with new values.
        
    Returns:
        plt.figure: The figure with the plot.
    """
    df_changed = df_changed.sort_values(by=new_col_name, ascending=True)
    df_changed.reset_index(inplace=True)
    
    bar_positions = np.arange(len(df_changed))
    bar_height = 0.4
    plot_width = 8
    fig, ax = plt.subplots(figsize=(plot_width, len(df_changed) / 4 + 2))

    ax.barh(bar_positions + bar_height, df_changed[new_col_name],\
            height=bar_height, color=NEW_GROUP_COLOR, label=new_col_name)

    ax.barh(bar_positions, df_changed[old_col_name], height=bar_height,\
        color=OLD_GROUP_COLOR, label=old_col_name)

    for i, diff in enumerate(df_changed[indicato_col]):
        x = max(df_changed[new_col_name][i], df_changed[old_col_name][i]) + 1
        y = bar_positions[i] + bar_height / 2
        ax.text(x, y, f'{diff}', va='center')

    ax.legend()
    ax.set_yticks(bar_positions + bar_height)
    ax.set_yticklabels(df_changed['Label'])

    plt.text(0.5, 1.3, s='Comparison Report',\
             transform=ax.transAxes, fontsize=22, verticalalignment='top',\
                horizontalalignment='center')
    plt.text(0.5, 1.1, 'The groups that have changed',\
            transform=ax.transAxes, fontsize=12, verticalalignment='top',\
                horizontalalignment='center')
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

    color = NEW_GROUP_COLOR if new_group else OLD_GROUP_COLOR
    plt.barh(df[label_col_name], df[vals_col_name], color=color)

    bar_positions = np.arange(len(df))

    logger.debug(f'bar_positions: {bar_positions}')

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


def plot_group(df,
               vals_col_name: str,
               indicator_value: str,
               columns_for_label: list,
               indicator_col_name: str,
               plot_title: str,
               no_topic_phrase: str
               ) -> plt.figure:

    logger.debug(df.head())

    logger.debug(f'indicator_col_name: {indicator_col_name}, indicator_value: {indicator_value},\
                  indicator_col_name: {indicator_col_name}')
    

    logger.debug(f'indicator value: {indicator_value}')
    logger.debug(f'indicator_col_name: {indicator_col_name}')


    df = df[columns_for_label + [vals_col_name, indicator_col_name]]


    df.where(df[str(indicator_col_name)] == 'Old group', inplace=True)
    
    df.dropna(inplace=True)

    df['Label'] = df[columns_for_label].apply(
        lambda x: combine_words(x, no_topic_phrase), axis=1
        )

    # df[vals_col_name] = pd.to_numeric(df[vals_col_name], errors='coerce')
    # df.dropna(inplace=True)
    df[vals_col_name] = df[vals_col_name].astype(int)
    df.reset_index(inplace=True)

    logger.debug(f'df: {df.head()}')

    df = df[['Label'] + [vals_col_name]]
 
    fig = create_single_bar_barh(df,
                                 vals_col_name=vals_col_name,
                                 label_col_name='Label',
                                 plot_title=plot_title)
    
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
            pdf.savefig(fig, bbox_inches='tight')
        plt.close()


def create_pdf_comaprison_report(
        df: pd.DataFrame,
        old_col_name: str,
        new_col_name: str,
        old_col_value: str,
        new_col_value: str,
        cols_for_label: list,
        cols_for_old_label: list,
        indicator_col_name: str,
        new_group_plot_title: str,
        old_group_plot_title: str,
        no_topic_phrase: str,
        output_file_path: str
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
                                             indicator_col_name)    
    
    fig_new_group = plot_group(df,
                               vals_col_name=old_col_name,
                               indicator_value=new_col_value,
                               columns_for_label=cols_for_label,
                               indicator_col_name=indicator_col_name,
                               plot_title=new_group_plot_title,
                               no_topic_phrase=no_topic_phrase)
    
    fig_old_group = plot_group(df,
                               vals_col_name=new_col_name,
                               indicator_value=old_col_value,
                               columns_for_label=cols_for_old_label,
                               indicator_col_name=indicator_col_name,
                               plot_title=old_group_plot_title,
                               no_topic_phrase=no_topic_phrase)

    table = create_table(df)

    logger.info('Creating comparison report PDF...')

    fig_list = [fig_changed_groups, fig_new_group, fig_old_group, table]
    logger.debug(f'{fig_list}')

    save_to_pdf(fig_list,
                output_file_path)
    
    logger.info(f'Comparison report PDF was saved to {output_file_path}')
