import peppi_py

from slippi_db.parse_peppi import get_buttons

game = peppi_py.read_slippi('./runback_replay.slp')

# get the last frame
last_frames = game.frames[-1:]

port_names = sorted(p['port'] for p in game.start['players'])
ports_data = game.frames[-1:].field('ports')
for i, port_name in enumerate(port_names):
    # print the last frame
    for frame in last_frames:
        player = ports_data.field(port_name)
        leader = player.field('leader')
        pre = leader.field('pre')

        buttons = get_buttons(pre.field('buttons_physical').fill_null(0))
        print(buttons)
