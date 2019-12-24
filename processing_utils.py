import re
import json
import argparse
import jsonlines
import pickle as pkl
import datetime as dt
import lyricsgenius
from lyricsgenius.api import *
from lyricsgenius.utils import sanitize_filename
from genius_keys import GENIUS_ACCESS_TOKEN
from collections import defaultdict


RETOK = re.compile(r'\w+|[^\w\s\[\]]|\n', re.UNICODE)
SONG_PART_REGEX = re.compile(r'\[\w+\W?\s?\w*\d?\]', re.UNICODE)

def read_json(file_path):
    with open(file_path, 'r') as f:
        json_file = json.load(f)
    return json_file

def write_jsonl(file_path, dict_obj):
    with jsonlines.open(file_path, 'w') as f:
        f.write_all(dict_obj)

def create_counts_dict(json_file, regex):
    '''
    function for determining frequency of occurrence 
    for each word
    '''
    word_count_dict = defaultdict(int)
    for song in json_file['songs']:
        for word in regex.findall(song['lyrics']):
            word_count_dict[word.lower()] += 1
    return word_count_dict

def get_lyrics_from_json(json_file, regex):
    '''
    iterate through Genius JSON file and retrieve
    lyrics. remove song-part designation
    '''
    all_lyrics = []
    for song in json_file['songs']:
        lyrics = regex.sub('', song['lyrics'])
        lyrics = lyrics.lower()
        all_lyrics.append(lyrics)
    return all_lyrics

def create_text_and_target(songs, num_songs, split=0.8):
    split_idx = int(num_songs * 0.8)
    train_text_and_targets = []
    valid_text_and_targets = []

    for idx, song in enumerate(songs):
        song = song.split('\n')
        # 0-index refers to song-part designation, i.e. [INTRO],
        # that was replaced with empty string
        base_lyric = song[1]
        for lyric in song[2:]:
            text_and_target = {}
            if lyric:
                if idx < split_idx:
                    text_and_target['text'] = base_lyric
                    text_and_target['labels'] = [lyric]
                    train_text_and_targets.append(text_and_target)
                else:
                    text_and_target['text'] = base_lyric
                    text_and_target['eval_labels'] = [lyric]
                    valid_text_and_targets.append(text_and_target)

                base_lyric += ('\n ' + lyric)

    write_jsonl('./train_lyrics.jsonl', train_text_and_targets)
    write_jsonl('./valid_lyrics.jsonl', valid_text_and_targets)

# rewriting / overwriting some of the existing functions in lyricsgenius package
def find_artist_id(search_term, genius_obj):
    if genius_obj.verbose:
        print('Retrieving artist ID for {0}...\n'.format(search_term))

    # Perform a Genius API search for the artist
    found_artist = None
    response = genius_obj.search_genius_web(search_term)
    found_artist = genius_obj._get_item_from_search_response(response, type_="artist")

    # Exit the search if we couldn't find an artist by the given name
    if not found_artist:
        if genius_obj.verbose:
            print("No results found for '{a}'.".format(a=search_term))
        return None

    # Assume the top search result is the intended artist
    return found_artist['id']

def force_search_artist(genius_obj, artist_name, sleep_time=600, max_songs=None,
                  sort='popularity', per_page=20,
                  get_full_info=True,
                  allow_name_change=True):
    #     repurposed from lyricsgenius package
    """Search Genius.com for songs by the specified artist.
    Returns an Artist object containing artist's songs.
    :param artist_name: Name of the artist to search for
    :param max_songs: Maximum number of songs to search for
    :param sort: Sort by 'title' or 'popularity'
    :param per_page: Number of results to return per search page
    :param get_full_info: Get full info for each song (slower)
    :param allow_name_change: (bool) If True, search attempts to
                              switch to intended artist name.
    """
    artist_id = find_artist_id(artist_name, genius_obj)
    if artist_id == None:
        return None

    artist_info = genius_obj.get_artist(artist_id)
    found_name = artist_info['artist']['name']
    if found_name != artist_name and allow_name_change:
        if genius_obj.verbose:
            print("Changing artist name to '{a}'".format(a=found_name))
        artist_name = found_name

    # Create the Artist object
    artist = Artist(artist_info)

    # Download each song by artist, stored as Song objects in Artist object
    page = 1
    reached_max_songs = False
    
    # keep track of song IDs already encountered to avoid re-adding in the event of an API timeout
    collected_song_ids = []
    
    while not reached_max_songs:
        try:
            songs_on_page = genius_obj.get_artist_songs(artist_id, sort, per_page, page)

            # Loop through each song on page of search results
            for song_info in songs_on_page['songs']:
                if song_info['id'] not in collected_song_ids:
                    collected_song_ids.append(song_info['id'])
                    # Check if song is valid (e.g. has title, contains lyrics)
                    has_title = ('title' in song_info)
                    has_lyrics = genius_obj._result_is_lyrics(song_info['title'])
                    valid = has_title and (has_lyrics or (not genius_obj.skip_non_songs))

                    # Reject non-song results (e.g. Linear Notes, Tracklists, etc.)
                    if not valid:
                        if genius_obj.verbose:
                            s = song_info['title'] if has_title else "MISSING TITLE"
                            print('"{s}" is not valid. Skipping.'.format(s=s))
                        continue

                    # Create the Song object from lyrics and metadata
                    lyrics = genius_obj._scrape_song_lyrics_from_url(song_info['url'])
                    if get_full_info:
                        info = genius_obj.get_song(song_info['id'])
                    else:
                        info = {'song': song_info}
                    song = Song(info, lyrics)


                    # Attempt to add the Song to the Artist
                    result = artist.add_song(song, verbose=False)
                    if result == 0 and genius_obj.verbose:
                        print('Song {n}: "{t}"'.format(n=artist.num_songs,
                                                       t=song.title))

                    # Exit search if the max number of songs has been met
                    reached_max_songs = max_songs and artist.num_songs >= max_songs
                    if reached_max_songs:
                        if genius_obj.verbose:
                            print('\nReached user-specified song limit ({m}).'.format(m=max_songs))
                        break
                else:
                    continue
        # temporarily stop API requests when encountering a timeout or when
        # there's 
        except (requests.exceptions.ReadTimeout, TypeError) as e:
            print('Encountered the following error: ')
            print(e)
            print('Sleeping for {} seconds... '.format(sleep_time))
            time.sleep(sleep_time)

        # Move on to next page of search results
        page = songs_on_page['next_page']
        if page is None:
            break  # Exit search when last page is reached

    if genius_obj.verbose:
        print('Done. Found {n} songs.'.format(n=artist.num_songs))
    return artist

def get_and_process_songs(args):
    stock_filename = 'Lyrics_' + args.artist_name.replace(' ', '') + '.json'
    stock_filename = sanitize_filename(stock_filename)

    if args.download_lyrics:
        start = dt.datetime.now()
        print('{}| Beginning download'.format(start))
        genius = lyricsgenius.Genius(GENIUS_ACCESS_TOKEN, sleep_time=1)
        artist_tracks = force_search_artist(genius, args.artist_name, sleep_time=600, max_songs=None,
                                             sort='popularity', per_page=20,
                                             get_full_info=True,
                                             allow_name_change=True)
        print('{}| Finished download in {}'.format(dt.datetime.now(), (dt.datetime.now() - start)))
        artist_tracks.save_lyrics()

    genius_file = read_json('./'+stock_filename)    
    word_counts = create_counts_dict(genius_file, RETOK)
    pkl.dump(word_counts, open('word_counts_dict.p','wb'))
    artist_lyrics = get_lyrics_from_json(genius_file, SONG_PART_REGEX)
    create_text_and_target(artist_lyrics, len(artist_lyrics))


        '''
        TO ADD
        AFTER WRITING OR IN CASE OF LOADING JSON LYRICS FILE, SPECIFY FILE PATH.
        1) FORMAT COUNT DICT SAME AS DICT INPUT FOR ChatDictionary() CLASS
        2) CREATE DATASET() CLASS OBJECT; i.e. instance of ChatDataset() CLASS
        2.5) ADD ABOVE 2 AS PRE-SAVED FILES FOR SKIPPING DOWNLOAD/PREPROCESS SET???
        3) TRAINING MODEL PORTION
        '''