import re
import os
import glob
import json
import argparse
import jsonlines
import pickle as pkl
import lyricsgenius
from lyricsgenius.api import *
from collections import defaultdict
from genius_keys import GENIUS_ACCESS_TOKEN

RETOK = re.compile(r'\w+|[^\w\s\[\]]|\n', re.UNICODE)
SONG_PART_REGEX = re.compile(r'\[\w+\W?\s?\w*\d?\]', re.UNICODE)
SPECIAL_CHARACTERS = ['__null__', '__start__', '__end__', '__unk__']

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
    word_count_dict['__null__'] +=  1000000003
    word_count_dict['__start__'] += 1000000002
    word_count_dict['__end__'] +=   1000000001
    word_count_dict['__unk__'] +=   1000000000
    for song in json_file['songs']:
        for word in regex.findall(song['lyrics']):
            word_count_dict[word.lower()] += 1
    return word_count_dict

def combine_counts(counts_list):
    '''
    function for combining pickled word counts. only applies 
    when training chatbot that uses lyrics from more than
    one artist. see text_and_target_from_dir()
    '''
    counts = {}
    for count in counts_list:
        for word in count:
            if word in counts.keys():
                if word in SPECIAL_CHARACTERS:
                    continue
                else:
                    counts[word] += count[word]
            else:
                counts[word] = count[word]
    pkl.dump(counts, open('word_counts_mixed_dict.p', 'wb'))

def to_exclude(word):
    '''
    remove troublesome characters/words from lyrics
    '''
    blacklisted = [' ', '', '\'', '\"', '\xa0', '\n', '\t']
    blacklisted += ['.', ]
    whitespace = re.compile(r'^\s{2,}')
    if word not in blacklisted and not whitespace.search(word):
        return True
    else:
        return False

def get_lyrics_from_json(json_file, regex):
    '''
    iterate through Genius JSON file and retrieve
    lyrics. attempt to remove song-part designation,
    e.g. [INTRO] or [CHORUS]
    '''
    songs = []
    all_lyrics = []
    for song in json_file['songs']:
        lyrics = regex.sub('', song['lyrics'])
        lyrics = lyrics.lower()
        songs.append(lyrics)
    for song in songs:
        all_lyrics += list(filter(to_exclude, song.split('\n')))
    return all_lyrics

def create_text_and_target(lyrics, lookback, split=0.8, write=True):
    split_idx = int(len(lyrics) * 0.8)
    train_text_and_targets = []
    valid_text_and_targets = []
    
    train_lyrics = lyrics[:split_idx]
    valid_lyrics = lyrics[split_idx:]

    for i in range(lookback, len(train_lyrics)):
        text_and_target = {}
        train_text = " ".join(train_lyrics[i-lookback:i])
        train_label = train_lyrics[i]
        if not train_text.isspace() and not train_label.isspace():
            text_and_target['text'] = train_text
            text_and_target['labels'] = train_label
            train_text_and_targets.append(text_and_target)

    for i in range(lookback, len(valid_lyrics)):
        text_and_target = {}
        valid_text = " ".join(valid_lyrics[i-lookback:i])
        valid_label = valid_lyrics[i]
        if not valid_text.isspace() and not valid_label.isspace():
            text_and_target['text'] = valid_text
            text_and_target['eval_labels'] = valid_label
            valid_text_and_targets.append(text_and_target)
    if write:
        write_jsonl('./train_lyrics.jsonl', train_text_and_targets)
        write_jsonl('./valid_lyrics.jsonl', valid_text_and_targets)
    else:
        return train_text_and_targets, valid_text_and_targets

def text_and_target_from_dir(path, lookback):
    train_lyrics = []
    valid_lyrics = []
    counts = []
    for file in glob.glob(os.path.join(path, '*.json')):
        json_file = read_json(file)
        raw_lyrics = get_lyrics_from_json(json_file, SONG_PART_REGEX)
        counts_ = create_counts_dict(json_file, RETOK)
        train_, valid_ = create_text_and_target(raw_lyrics, lookback=lookback, write=False)
        train_lyrics += train_
        valid_lyrics += valid_
        counts.append(counts_)
    combine_counts(counts)
    write_jsonl('./train_lyrics_mixed.jsonl', train_lyrics)
    write_jsonl('./valid_lyrics_mixed.jsonl', valid_lyrics)

def find_artist_id(search_term, genius_obj):
    # rewriting / overwriting some of the existing functions in lyricsgenius package
    '''
    Function for simply returning aassociated artist_id
    :param search_term: Artist name to search for
    :param genius_obj: lyricgenius object instance used to make API call
    '''
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

def force_search_artist(genius_obj, artist_name, sleep_time=180, max_songs=None,
                  sort='popularity', per_page=20, get_full_info=True,
                  allow_name_change=True):
    ###    repurposed from lyricsgenius package         ###
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
        # there's an error
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
