from fire import Fire
import tile 
import thumbnail



if __name__ == '__main__':
    Fire({
        'thumbnail': thumbnail.Thumbnail,
        'tile': tile.Tile,
    })