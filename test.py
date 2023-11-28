import youtube_url_to_heatmap_coordinates as heatmap

url = ["https://www.youtube.com/watch?v=gtvWiwvJT3M","https://www.youtube.com/watch?v=2Wuoqt29e2A","https://www.youtube.com/watch?v=3-BZxvKT6x4"]

for url in url:
    print(heatmap.youtube_url_to_heatmap_coordinates(url))