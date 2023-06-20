import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

color = ['#E966A0','#2B2730','#6554AF','#9575DE','#0E2954','#1F6E8C','#2E8A99','#84A7A1']

# Membaca data dari file CSV
dataset1 = pd.read_csv('C:/Users/Reitama/visdat/Hotel Reservations.csv')

# Melakukan pengelompokkan berdasarkan keuntungan (booking not cancelled)
dataset2 = dataset1[dataset1['booking_status'] == 'Not_Canceled']

# Data untuk pemilik hotel
pemilik_hotel = dataset2.drop(['repeated_guest', 'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled',
                              'required_car_parking_space', 'lead_time', 'no_of_special_requests'], axis='columns')

# Data untuk First Officer
front_officer = dataset2.drop(['market_segment_type','avg_price_per_room','no_of_adults','no_of_children','no_of_weekend_nights','no_of_week_nights','repeated_guest','no_of_previous_cancellations','no_of_previous_bookings_not_canceled', 'required_car_parking_space', 'lead_time', 'no_of_special_requests', 'booking_status'], axis = 'columns')

# Data untuk Marketing 
marketing = dataset2[['Booking_ID', 'market_segment_type', 'room_type_reserved', 'arrival_year', 'arrival_month', 'arrival_date']]

TARGET = 'room_type_reserved'

def select_columns(dataset: pd.DataFrame, threshold: int = 10) -> list[str]:
    """
    Function used to divide the columns of a DataFrame in discrete and continuous based on the specified threshold.

    Args:
        df (pd.DataFrame): A DataFrame containing the columns to be selected.
        threshold (int): A threshold defining the minimum number of unique values. Default is 10.

    Returns:
        discrete_columns (list[str]): A list of discrete column names.
        continuous_columns (list[str]): A list of continuous column names.
    """
    
    # Initialize the discrete columns list
    discrete_columns = []

    for column in dataset.columns:
        # Select the columns with number of unique values less than or equal to the threshold
        if dataset[column].nunique() <= threshold:
            # Append the selected columns to the list of discrete columns
            discrete_columns.append(column)
    
    # Select the continuous columns
    continuous_columns = [col for col in dataset.columns.tolist() if col not in discrete_columns]


    return discrete_columns, continuous_columns

discrete_columns, continuous_columns = select_columns(dataset1)

def visualizations_discrete(dataset: pd.DataFrame):
    """
    Function used to plot countplots for each discrete column.

    Parameters:
        - df (pd.DataFrame): A DataFrame containing the data.
    """

    fig, axes = plt.subplots(nrows=len(discrete_columns) * 2, ncols=1, figsize=(10, 5 * len(discrete_columns) * 2))
    fig.subplots_adjust(hspace=0.3)
    sns.color_palette("Blues", as_cmap=True)

    for index, column in enumerate(discrete_columns):
        # Plot countplot
        sns.countplot(data=dataset, x=column, ax=axes[index * 2], palette=color)
        axes[index * 2].set_title(f"Countplot of {column}")
        axes[index * 2].set_xlabel("")

        # Plot countplot
        sns.countplot(data=dataset, x=column, hue=TARGET, ax=axes[index * 2 + 1], palette=color)
        axes[index * 2 + 1].set_title(f"Countplot of {column} specified by {TARGET}")
        axes[index * 2 + 1].set_xlabel("")

    # Show the plot
    st.pyplot(fig)

# Halaman Utama
def main():
    st.title('Data Utama')
    st.write('Konten Halaman Utama')
    st.write('Data Awal:')
    st.write(dataset1)
    st.write('Data setelah Pengelompokkan:')
    st.write(dataset2)
    visualizations_discrete(dataset1)

# Sub-halaman 1
def subpage1():
    st.title('Owner')
    st.write('Data Owner:')
    st.write(pemilik_hotel) # membaca data pada pemilik hotel
    pemilik_hotel_sortBy_roomType = pemilik_hotel[['room_type_reserved', 'avg_price_per_room']].groupby(by='room_type_reserved').sum()
    pemilik_hotel_year2017 = pemilik_hotel[pemilik_hotel['arrival_year'] == 2017]
    pemilik_hotel_groupBy_Month2017 = pemilik_hotel_year2017[['arrival_month', 'avg_price_per_room']].groupby(by='arrival_month').sum()
    pemilik_hotel_year2018 = pemilik_hotel[pemilik_hotel['arrival_year'] == 2018]
    pemilik_hotel_groupBy_Month2018 = pemilik_hotel_year2018[['arrival_month', 'avg_price_per_room']].groupby(by='arrival_month').sum()
    # Membuat plot grafik
    st.write('Bar Chart Keuntungan berdasarkan Tipe ruangan')
    st.bar_chart(pemilik_hotel_sortBy_roomType)
    
    fig, ax = plt.subplots()
    ax.plot(pemilik_hotel_groupBy_Month2017.index, pemilik_hotel_groupBy_Month2017.values)
    plt.suptitle('Grafik keuntungan tahun 2017')
    plt.xlabel('Bulan')
    plt.ylabel('Total keuntungan')
    st.pyplot(fig)

    fig2, ax2 = plt.subplots()
    ax2.plot(pemilik_hotel_groupBy_Month2018.index, pemilik_hotel_groupBy_Month2018.values)
    plt.suptitle('Grafik keuntungan tahun 2018')
    plt.xlabel('Bulan')
    plt.ylabel('Total keuntungan')
    st.pyplot(fig2) 


    
# Sub-halaman 2
def subpage2():
    st.title('Front Officer')
    st.write('Data Front Officer:')
    st.write(front_officer) # membaca data yang digunakan pada front officer
    # front_officer_sortRoomType_byBookingID = front_officer.drop(['arrival_year', 
    #                                                            'arrival_month',
    #                                                            'arrival_date', 'type_of_meal_plan'], axis = 'columns').groupby(by='room_type_reserved').count()
    front_officer_year2017 = front_officer[front_officer['arrival_year'] == 2017]
    front_officer_year2018 = front_officer[front_officer['arrival_year'] == 2018]
    front_officer_year2017_groupBy_roomType = front_officer_year2017[['Booking_ID', 'room_type_reserved']].groupby(by='room_type_reserved').count()
    front_officer_year2018_groupBy_roomType = front_officer_year2018[['Booking_ID', 'room_type_reserved']].groupby(by='room_type_reserved').count()
    overall_room_type_reserved = front_officer[['room_type_reserved', 'Booking_ID']].groupby(by='room_type_reserved').count()
    overall_type_of_meal_plan = front_officer[['type_of_meal_plan', 'Booking_ID']].groupby(by='type_of_meal_plan').count()
    st.write('Best sale Kamar Hotel 2017')
    st.bar_chart(front_officer_year2017_groupBy_roomType)
    st.write('Best sale Kamar Hotel 2018')
    st.bar_chart(front_officer_year2018_groupBy_roomType)
    st.write('Best seller Kamar')
    st.bar_chart(overall_room_type_reserved)
    st.write('Best Seller Paket Makanan')
    st.bar_chart(overall_type_of_meal_plan)

def marketings():
    st.title('Marketing')
    st.write('Data Marketing:')
    st.write(marketing) #membaca data marketing
    marketing_year2017 = marketing[marketing['arrival_year']==2017]
    marketing_year2017_groupBy = marketing_year2017[['market_segment_type', 'Booking_ID']].groupby(by='market_segment_type').count()
    marketing_year2018 = marketing[marketing['arrival_year']==2018]
    marketing_year2018_groupBy = marketing_year2018[['market_segment_type', 'Booking_ID']].groupby(by='market_segment_type').count()
    st.write('Tipe Market yang terlaris tahun 2017')
    st.bar_chart(marketing_year2017_groupBy)
    st.write('Best Seller Paket Makanan')
    st.bar_chart(marketing_year2018_groupBy)

    colunas = dataset1['arrival_month'].value_counts()

    plt.figure(figsize=(12,40))
    for i, col in enumerate(colunas.index):
        ax = plt.subplot(12,2,i+1)
        sns.barplot(x='market_segment_type',
                    y='avg_price_per_room', palette=color,
                    hue='booking_status', data=dataset1[dataset1['arrival_month'] == col])
        plt.title(f'Bulan {col}',fontweight='bold')
    plt.tight_layout()
    st.pyplot()

# Sidebar Navigation
pages = {
    
    'Data': main,
    'Owner': subpage1,
    'Front Officer': subpage2,
    'Marketing': marketings

}

# Menampilkan sidebar
st.sidebar.title('Pilih tipe akun anda:')
selection = st.sidebar.radio('Pilih Halaman', list(pages.keys()))

# Memanggil halaman yang dipilih
page = pages[selection]
page()