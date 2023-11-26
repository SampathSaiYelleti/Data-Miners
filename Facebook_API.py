import requests

def get_facebook_likes_count(post_url, access_token):
    # Extract post ID from the post URL
    post_id = post_url.split('/')[-1]

    # Make a request to the Facebook Graph API
    api_url = f'https://graph.facebook.com/v18.0/{post_id}?fields=likes.summary(true)&access_token={access_token}'
    response = requests.get(api_url)

    if response.status_code == 200:
        data = response.json()
        likes_count = data['likes']['summary']['total_count']

        return likes_count
    else:
        data = response.json()
        print(data)
        print(f"Error: {response.status_code}")
        return None


access_token = 'EAAFAwm07nw4BO1ZCPCyi0rqXQuZCPYA9upreAPmVU73MLzeRZC5Qd9o79CzJi2kXjlAATZB6jZBrVy99ZCjl6uAOVuuv24c2dGmPLS9xLhG2smGlWK13cUohtucZCPUciiMcnBMwVZBdzSZCM4Lm2KufV9QO8SH2GaxESmFfwJsCDau9UAUcwh1iMO2PkzZCEQWhNgn6ZCDZC9KC5s1Qx5P1x0hKZA1265XZB7zCYRGQKqvFC6eIn9QKMchjkqSivJXqlwBjIZD'
post_url = 'https://www.facebook.com/facebook/posts/10157006965041729'

likes_count = get_facebook_likes_count(post_url, access_token)

if likes_count is not None:
    print(f"Likes: {likes_count}")
