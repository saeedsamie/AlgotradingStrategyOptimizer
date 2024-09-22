import pandas as pd


def process_trades_file(input_file_path, output_file_path):
    # Load the trades CSV file
    df = pd.read_csv(input_file_path)

    # Strip any extra spaces from the column names
    df.columns = df.columns.str.strip()

    # Convert the 'time' column to datetime format
    df['time'] = pd.to_datetime(df['time'], format='%m/%d/%Y %H:%M')

    # Separate the entry and exit trades for both long and short positions
    entry_long_trades = df[df['type'] == 'Entry long'].reset_index(drop=True)
    exit_long_trades = df[df['type'] == 'Exit long'].reset_index(drop=True)

    entry_short_trades = df[df['type'] == 'Entry short'].reset_index(drop=True)
    exit_short_trades = df[df['type'] == 'Exit short'].reset_index(drop=True)

    # Align entry and exit times for long trades
    min_length_long = min(len(entry_long_trades), len(exit_long_trades))
    entry_long_trades = entry_long_trades.iloc[:min_length_long]
    exit_long_trades = exit_long_trades.iloc[:min_length_long]

    # Align entry and exit times for short trades
    min_length_short = min(len(entry_short_trades), len(exit_short_trades))
    entry_short_trades = entry_short_trades.iloc[:min_length_short]
    exit_short_trades = exit_short_trades.iloc[:min_length_short]

    # Combine the data for long trades
    entry_long_trades['Profit'] = exit_long_trades['profit']

    # Combine the data for short trades
    entry_short_trades['Profit'] = exit_short_trades['profit']

    # Concatenate both long and short trades
    combined_trades = pd.concat([entry_long_trades[['time', 'Profit']],
                                 entry_short_trades[['time', 'Profit']]])

    # Convert the 'Profit' column to 1 (profitable) or 0 (non-profitable)
    combined_trades['Profit'] = combined_trades['Profit'].apply(lambda x: 1 if x > 0 else 0)

    # Save the resulting dataset to a new CSV file
    combined_trades.to_csv(output_file_path, index=False)
    print(f"Processed dataset saved to {output_file_path}")


if __name__ == "__main__":
    # Example usage:
    process_trades_file("RSI_5m_1D_all.csv", "processed_trades.csv")
