import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def process_data(trades_file_path, onchain_file_path, output_file_path):
    # Step 1: Load trades and on-chain data
    trades_df = pd.read_csv(trades_file_path)
    onchain_df = pd.read_csv(onchain_file_path)

    # Step 2: Clean column names of on-chain data
    onchain_df.columns = onchain_df.columns.str.strip().str.replace(r'\(1\)|\(1\)\.csv|\.csv', '', regex=True)
    onchain_df.columns = onchain_df.columns.str.replace('1h', '').str.replace('-btc-', '')
    onchain_df = onchain_df[onchain_df.columns.drop(['c', 'h', 'l', 'o'])]
    trades_df.rename(columns={'profit ': 'profit'}, inplace=True)

    # Step 3: Normalize the on-chain data (excluding 'timestamp')
    scaler = MinMaxScaler()
    onchain_df[onchain_df.columns.drop('timestamp')] = scaler.fit_transform(
        onchain_df[onchain_df.columns.drop('timestamp')])

    # Step 4: Clean trades data (restore price column if removed, and ensure timestamps are in correct format)
    trades_df['time'] = pd.to_datetime(trades_df['time'])
    onchain_df['timestamp'] = pd.to_datetime(onchain_df['timestamp'])

    # Step 5: Fix timezone mismatches if they exist
    # Removing timezone info if present in 'timestamp' or 'time' columns
    trades_df['time'] = trades_df['time'].dt.tz_localize(None)
    onchain_df['timestamp'] = onchain_df['timestamp'].dt.tz_localize(None)

    # Step 6: Merge the trades data with the on-chain data using forward fill for values within the same hour
    merged_data = pd.merge_asof(trades_df.sort_values('time'),
                                onchain_df.sort_values('timestamp'),
                                left_on='time', right_on='timestamp',
                                direction='backward')

    # Step 7: Drop unnecessary columns and finalize dataset
    merged_data = merged_data.drop(
        columns=['timestamp', 'price', 'time', 'price', 'profit percent', 'signl', 'type']).reset_index(drop=True)

    # Step 8: Convert 'Profit' column to binary (1 if profit > 0, else 0)

    merged_data['profit'] = merged_data['profit'].apply(lambda x: 1 if x > 0 else 0)
    # Step 9: Save the final dataset to the output file
    merged_data.to_csv(output_file_path, index=False)
    print(f"Processed dataset saved to {output_file_path}")


if __name__ == "__main__":
    # Example usage:
    process_data("RSI_5m_1D_all.csv", "FinalDatasetBestFeatures.csv", "final_merged_dataset.csv")
