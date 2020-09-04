import sys
import pickle as pkl
from args import get_setup_args
from processing_utils import *
from lyricsgenius.utils import sanitize_filename

def get_and_process_songs(args):
    if args.artist_name is None and args.download_lyrics:
        print('\n********** ERROR *************')
        print('Invalid or missing artist name.')
        print('\n******************************\n')
        sys.exit()

    if args.download_lyrics:
        start = dt.datetime.now()
        stock_filename = 'Lyrics_' + args.artist_name.replace(' ', '') + '.json'
        stock_filename = sanitize_filename(stock_filename)
        print('{}| Beginning download'.format(start))
        genius = lyricsgenius.Genius(GENIUS_ACCESS_TOKEN, sleep_time=1)
        artist_tracks = force_search_artist(genius, args.artist_name, sleep_time=600,
                                            max_songs=None,
                                            sort='popularity', per_page=20,
                                            get_full_info=True,
                                            allow_name_change=True)
        print('{}| Finished download in {}'.format(dt.datetime.now(), (dt.datetime.now() - start)))
        artist_tracks.save_lyrics()

    if args.load_path:
        genius_file = read_json(args.load_path)
    else:
        genius_file = read_json('./'+stock_filename)

    word_counts = create_counts_dict(genius_file, RETOK)
    pkl.dump(word_counts, open('word_counts_dict.p', 'wb'))
    artist_lyrics = get_lyrics_from_json(genius_file, SONG_PART_REGEX)
    create_text_and_target(artist_lyrics, len(artist_lyrics))

if __name__ == "__main__":
    get_and_process_songs(get_setup_args())
