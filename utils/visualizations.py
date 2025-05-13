# ============================IMPORT LIBRARIES============================
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg


def print_header(content, padding=10, width=None, tabs=False):
    """
    Prints a header wrapped inside a box.

    Parameters:
    - content: Text to display as header (automatically uppercase)
    - padding: Spaces on each side of the content inside the box
    - width: Optional fixed width for the entire box. If None, adjusts to content size.
    - tabs: If true, adds the spacing `\t`to center of the box
    """
    content = content.upper()
    inner_width = len(content) + (padding * 2)
    box_width = width if width and width > inner_width else inner_width

    top_bottom = "+" + "-" * (box_width + 2) + "+"
    padded_content = "| " + content.center(box_width) + " |"
    prefix = "\t" * 3 if tabs else ""

    print(f"{prefix}{top_bottom}")
    print(f"{prefix}{padded_content}")
    print(f"{prefix}{top_bottom}")


def visualize_images(df, num_images=5, title=None, figsize=(15, 3)):
    """
    Visualize images from dataset

    Parameters:
    - df: DataFrame metadata
    - num_images: Number of images to display
    - title: Optional title for the entire figure
    - figsize: Size of the figure as (width, height) tuple
    """

    # Get image paths from the DataFrame
    image_paths = df["image_path"].tolist()
    num_images = len(image_paths)

    fig, axes = plt.subplots(1, num_images, figsize=figsize, facecolor="white")

    # Handle case of a single image
    if num_images == 1:
        axes = [axes]

    for idx, (_, row) in enumerate(df.iterrows()):
        file_path = row["image_path"]

        # Build title based on metadata
        title_parts = [f"ID: {row['image_id']}"]
        for col in row.index:
            if col not in ["image_id", "image_path"]:
                title_parts.append(f"{col}: {row[col]}")
        img_title = "\n".join(title_parts)

        try:
            img = mpimg.imread(file_path)
            axes[idx].imshow(img)
            axes[idx].axis("off")
            axes[idx].set_title(img_title, fontsize=9)
        except FileNotFoundError:
            print(f"Image not found: {file_path}")
            axes[idx].text(0.5, 0.5, "Image not found", ha="center", va="center")
            axes[idx].axis("off")

    # Add overall title if availabel
    if title:
        plt.suptitle(title, fontsize=14, y=1.01)
        plt.subplots_adjust(top=0.85)

    plt.tight_layout()
    plt.show()


def visualize_category_distribution(df, column_name):
    """
    Visualizes the distribution of categories from a specified column in the provided DataFrame.

    Parameters:
    - df: pandas DataFrame containing the column to visualize.
    - column_name: The column name whose distribution will be visualized.
    """
    # Calculate the frequency of each category in the specified column
    category_counts = df[column_name].value_counts()

    # Create a figure with two side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(20, 10), facecolor="white")

    # Plot a pie chart in the first subplot
    color_palette = sns.color_palette("coolwarm")
    sns.set_palette(color_palette)
    axes[0].pie(
        category_counts, labels=category_counts.index, autopct="%1.1f%%", startangle=140
    )
    axes[0].set_title(f"{column_name} Distribution")

    # Plot a bar chart in the second subplot with the hue parameter
    sns.barplot(
        x=category_counts.index,
        y=category_counts.values,
        ax=axes[1],
        hue=category_counts.index,
        palette="coolwarm",
        legend=False,
    )
    axes[1].set_title(f"{column_name} Counts")

    # Adjust layout to ensure no overlapping of elements
    plt.tight_layout()

    # Display the plots
    plt.show()


def plot_grouped_bar(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    agg_func: str = "mean",
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    sort: bool = True,
    ascending: bool = False,
    palette: str = "coolwarm",
    figsize: tuple = (10, 6),
):
    """
    Creates a bar plot of the aggregated value of a numeric column grouped by a categorical column.

    Parameters:
    - df: pandas DataFrame
    - group_col: column to group by (categorical)
    - value_col: column to aggregate (numeric)
    - agg_func: aggregation function ('mean', 'sum', 'count', etc.)
    - title: chart title
    - x_label: label for x-axis
    - y_label: label for y-axis
    - sort: whether to sort the result
    - ascending: sorting order
    - palette: seaborn color palette
    - figsize: tuple indicating figure size
    """

    # Group and aggregate
    grouped = df.groupby(group_col)[value_col].agg(agg_func)

    if sort:
        grouped = grouped.sort_values(ascending=ascending)

    # Plot
    plt.figure(figsize=figsize)
    sns.barplot(
        x=grouped.values,
        y=grouped.index,
        hue=grouped.index,
        dodge=False,
        legend=False,
        palette=palette,
    )
    plt.title(title or f"{agg_func.capitalize()} of {value_col} by {group_col}")
    plt.xlabel(x_label or f"{agg_func.capitalize()} {value_col}")
    plt.ylabel(y_label or group_col)
    plt.tight_layout()
    plt.show()


def plot_grouped_bar(
    df,
    group_col,
    value_col,
    agg_func="mean",
    title="",
    x_label="",
    y_label="",
    sort=True,
    ascending=False,
    palette="coolwarm",
    figsize=(10, 6),
):
    """
    Creates a bar plot of the aggregated value of a numeric column grouped by a categorical column.

    Parameters:
    - df: pandas DataFrame
    - group_col: column to group by (categorical)
    - value_col: column to aggregate (numeric)
    - agg_func: aggregation function ('mean', 'sum', 'count', etc.)
    - title: chart title
    - x_label: label for x-axis
    - y_label: label for y-axis
    - sort: whether to sort the result
    - ascending: sorting order
    - palette: seaborn color palette
    - figsize: tuple indicating figure size
    """

    # Group and aggregate
    grouped = df.groupby(group_col)[value_col].agg(agg_func)

    if sort:
        grouped = grouped.sort_values(ascending=ascending)

    # Plot
    plt.figure(figsize=figsize)
    sns.barplot(
        x=grouped.values,
        y=grouped.index,
        hue=grouped.index,
        dodge=False,
        legend=False,
        palette=palette,
    )
    plt.title(title or f"{agg_func.capitalize()} of {value_col} by {group_col}")
    plt.xlabel(x_label or f"{agg_func.capitalize()} {value_col}")
    plt.ylabel(y_label or group_col)
    plt.tight_layout()
    plt.show()


def plot_grouped_heatmap(
    df,
    row_col,
    col_col,
    title="Grouped Heatmap",
    x_label="",
    y_label="",
    figsize=(12, 8),
):
    """
    Creates a heatmap showing the frequency of values at the intersection of two categorical columns.

    Parameters:
    - df: pandas DataFrame containing the data
    - row_col: Column name for the heatmap's rows (y-axis)
    - col_col: Column name for the heatmap's columns (x-axis)
    - title: Chart title
    - x_label: Label for x-axis (defaults to col_col name)
    - y_label: Label for y-axis (defaults to row_col name)
    - cmap: Matplotlib colormap name for the heatmap
    - figsize: Figure size as (width, height) tuple
    - fmt: Format string for annotations (e.g., 'd' for integers, '.1f' for floats)
    - annot: Whether to show value annotations on cells

    Returns:
    - fig, ax: The matplotlib figure and axis objects for further customization if needed
    """
    # Set default axis labels if not provided
    if not x_label:
        x_label = col_col
    if not y_label:
        y_label = row_col

    # Group and count the data
    grouped_counts = df.groupby([row_col, col_col]).size().unstack(fill_value=0)

    # Create the figure and plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the heatmap
    sns.heatmap(grouped_counts, annot=True, fmt="d", cmap="YlGnBu", cbar=True, ax=ax)

    # Add titles and labels
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)

    plt.tight_layout()
    plt.show()


def display_augmentation_examples(
    original_df,
    augmented_df,
    tiers,
    augmentations,
):
    """
    Display examples of each type of augmentation applied to images from different variety tiers

    Parameters:
    - original_df: original DataFrame metadata
    - augmented_df: augmented DataFrame metadata
    - tiers: Dictionary mapping tier names to lists of variety names belonging to that tier
    - augmentations: Dictionary mapping tier names to lists of augmentation types applied to that tier
    """
    for tier_name, varieties in tiers.items():
        if not varieties:
            continue

        # Take the first variety in this tier
        variety = varieties[0]

        # Find an example image for this variety
        sample_row = original_df[original_df["variety"] == variety].iloc[0]
        original_image_id = sample_row["image_id"]
        original_path = sample_row["image_path"]
        base_name = os.path.splitext(original_image_id)[0]

        print_header(f"Augmentation Examples for {tier_name} Tier - Variety: {variety}")

        # For each augmentation type applied to this tier
        aug_types = augmentations[tier_name]
        num_augs = len(aug_types)

        # Create a figure with 1 row for original + augmentations
        fig, axes = plt.subplots(
            1, num_augs + 1, figsize=(3 * (num_augs + 1), 4), facecolor="white"
        )

        # Display original image first
        try:
            original_img = mpimg.imread(original_path)
            axes[0].imshow(original_img)
            axes[0].set_title(f"Original", fontsize=10)
            axes[0].axis("off")
        except FileNotFoundError:
            axes[0].text(0.5, 0.5, "Image not found", ha="center", va="center")
            axes[0].axis("off")

        # Display each augmentation type
        for i, aug_type in enumerate(aug_types):
            aug_image_id = f"{base_name}_{aug_type}.jpg"

            aug_row = augmented_df[augmented_df["image_id"] == aug_image_id]

            aug_path = aug_row.iloc[0]["image_path"]

            try:
                aug_img = mpimg.imread(aug_path)
                axes[i + 1].imshow(aug_img)
                axes[i + 1].set_title(f"{aug_type}", fontsize=10)
                axes[i + 1].axis("off")
            except FileNotFoundError:
                axes[i + 1].text(
                    0.5, 0.5, "Augmented image\nnot found", ha="center", va="center"
                )
                axes[i + 1].axis("off")

        plt.suptitle(
            f"{tier_name} Tier ({variety}): Original vs Augmentations", fontsize=16
        )
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.show()
