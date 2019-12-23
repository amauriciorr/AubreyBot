import json
import pandas as pd
import argparse
import lyricsgenius
import datetime as dt
from lyricsgenius.api import *
from lyricsgenius.utils import sanitize_filename
from genius_keys import GENIUS_ACCESS_TOKEN
from utils import * 


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


def get_and_process_songs():
    parser = argparse.ArgumentParser('Download and pre-process rap lyrics')
    parser.add_argument("--artist_name",
                        type=str,
                        help="Name of rapper to get lyrics for.")
    parser.add_argument("--skip_download",
                        type=lambda s: s.lower().startswith('t'),
                        default=False,
                        help="If lyrics JSON file already created, avoid downloading and start with building vocab and tokenizing.")
    parser.add_argument("--load_path",
                        type=str,
                        default=None,
                        help="Path to load artist lyrics JSON.")
    args = parser.parse_args()

    stock_filename = 'Lyrics_' + args.artist_name.replace(' ', '') + '.json'
    stock_filename = sanitize_filename(stock_filename)

    if args.skip_download:

        genius_file = read_json('./'+stock_filename)
        '''
        TO ADD
        AFTER WRITING OR IN CASE OF LOADING JSON LYRICS FILE, SPECIFY FILE PATH.
        1) LOAD JSON
        2) CREATE COUNTS
        3) CREATE DICTIONARY() CLASS OBJECT
        4) CREATE TEXT-LABEL JSON FILE TO BE USED FOR NEXT STEP
        5) CREATE DATASET() CLASS OBJECT
        6) TRAINING MODEL PORTION
        '''

    else:
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
        


if __name__ == "__main__":
    get_and_process_songs()

