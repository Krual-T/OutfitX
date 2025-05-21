from typing import Optional


def get_outfit_index(location_id:Optional[str]= '101071404')->int:
    import requests
    API_HOST = 'mn5u9uyg32.re.qweatherapi.com'
    API_KEY = '1ec509cb419a44c38b99727694fe0532'
    weather_index_api = '/v7/indices/1d'
    param_to_str = lambda param: '&'.join([f'{k}={v}' for k, v in param.items()])
    param_dict = {
        'location': location_id,
        'type': '3',
        'key': API_KEY,
    }
    request_url = f'https://{API_HOST}{weather_index_api}/?{param_to_str(param_dict)}'
    resp = requests.get(request_url).json()
    return int(resp['daily'][0]['level'])