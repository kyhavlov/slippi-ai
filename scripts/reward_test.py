import numpy as np

from slippi_ai import reward, utils
from slippi_ai.types import Game, Player
from melee import enums

def create_mock_player(action, percent, stocks_left, character, frames, dead: bool = False):
    """Helper function to create a mock player."""
    return Player(
        action=action,
        percent=percent,
        x=np.zeros(frames),
        y=np.zeros(frames),
        facing=np.zeros(frames),
        jumps_left=np.zeros(frames),
        shield_strength=np.zeros(frames),
        on_ground=np.zeros(frames),
        is_dead=np.full(frames, dead),
        controller={
            'main_stick': np.zeros((frames, 2)),
            'c_stick': np.zeros((frames, 2)),
            'shoulder': np.zeros(frames),
            'buttons': np.zeros((frames, 5))
        },
        stocks_left=stocks_left,
        character=character,
        invulnerable=np.zeros(frames)
    )

def batch_players(*players):
    """
    Generic function to batch multiple players along a new batch dimension.
    All players must have the same number of frames.
    
    Args:
        *players: Player instances to batch
    Returns:
        A new Player with arrays batched along axis 1
    """
    # Get field names from the Player namedtuple
    player_fields = players[0]._fields
    result = {}
    
    for field_name in player_fields:
        field_value = getattr(players[0], field_name)
        
        if field_name == 'controller':
            # Handle controller dict specially
            controller_dict = {}
            for ctrl_key in field_value:
                # Stack each controller component
                controller_values = [p.controller[ctrl_key] for p in players]
                controller_dict[ctrl_key] = np.stack(controller_values, axis=1)
            result[field_name] = controller_dict
        else:
            # Stack regular fields
            field_values = [getattr(p, field_name) for p in players]
            result[field_name] = np.stack(field_values, axis=1)
    
    return Player(**result)

def batch_games(*games):
    """
    Generic function to batch multiple games along a new batch dimension.
    All games must have the same number of frames.
    
    Args:
        *games: Game instances to batch
    Returns:
        A new Game with arrays batched along axis 1
    """
    return Game(
        p0=batch_players(*[g.p0 for g in games]),
        p1=batch_players(*[g.p1 for g in games]),
        p2=batch_players(*[g.p2 for g in games]),
        p3=batch_players(*[g.p3 for g in games]),
        stage=np.stack([g.stage for g in games], axis=1),
        randall_phase=np.stack([g.randall_phase for g in games], axis=1),
        is_teams=np.stack([g.is_teams for g in games], axis=1)
    )

def compute_rewards_test():
    # Create mock player data for 3 frames
    p0 = create_mock_player(action=np.array([0xE, 0xE, 0x0]), percent=np.array([0, 50, 50]), stocks_left=np.array([4, 4, 3]), character=np.array([3, 3, 3]), frames=3)
    p1 = create_mock_player(action=np.array([0, 0, 0]), percent=np.array([0, 0, 0]), stocks_left=np.array([4, 4, 4]), character=np.array([0, 0, 0]), frames=3)
    p2 = create_mock_player(action=np.array([0, 0, 0]), percent=np.array([0, 0, 0]), stocks_left=np.array([4, 4, 4]), character=np.array([0, 0, 0]), frames=3)
    p3 = create_mock_player(action=np.array([0, 0, 0]), percent=np.array([0, 0, 0]), stocks_left=np.array([4, 4, 4]), character=np.array([0, 0, 0]), frames=3)

    # Create mock game
    game = Game(p0=p0, p1=p1, p2=p2, p3=p3, stage=np.array([0, 0, 0]), randall_phase=np.array([0, 0, 0]), is_teams=np.array([True, True, True]))

    # Compute rewards
    rewards = reward.compute_rewards(game, damage_ratio=0.01, ledge_grab_penalty=0.01, approaching_factor=0, stalling_penalty=0)

    # Print rewards for verification
    print("Rewards:", rewards)

    # create a mock singles game
    game2p0 = create_mock_player(action=np.array([0, 0, 0]), percent=np.array([0, 0, 0]), stocks_left=np.array([4, 4, 4]), character=np.array([4, 4, 4]), frames=3)
    game2p1 = create_mock_player(action=np.array([0, 0, 0]), percent=np.array([0, 0, 0]), stocks_left=np.array([0, 0, 0]), character=np.array([0, 0, 0]), frames=3)
    #game2p1 = game2p1._replace(x=np.array([1000, 1000, 1000]), y=np.array([1000, 1000, 1000]), is_dead=np.array([True, True, True]))
    game2p2 = create_mock_player(action=np.array([0, 0, 0]), percent=np.array([0, 10, 10]), stocks_left=np.array([0, 0, 0]), character=np.array([0, 0, 0]), frames=3, dead=True)
    game2p3 = create_mock_player(action=np.array([0xE, 0xE, 0x0]), percent=np.array([0, 20, 20]), stocks_left=np.array([4, 4, 3]), character=np.array([0, 0, 0]), frames=3)

    game2 = Game(p0=game2p0, p1=game2p1, p2=game2p2, p3=game2p3, stage=np.array([0, 0, 0]), randall_phase=np.array([0, 0, 0]), is_teams=np.array([False, False, False]))

    # To mimic how evaluators.Trajectory.batch works in the real RL code:
    # We need to create a batch of games with shape [T, B] where:
    # - T is time dimension (frames)
    # - B is batch dimension (different environments)
    
    # Create a properly batched game with all frames using our generic batch_games function
    batched_games = batch_games(game, game2)
    
    print(f"Properly batched games shape - p0.action: {batched_games.p0.action.shape}")
    print(f"Properly batched games is_teams shape: {batched_games.is_teams.shape}")
    
    # Compute rewards for the properly batched games
    # This should produce rewards with shape [T-1, B] = [2, 2]
    rewards_batched = reward.compute_rewards(
        batched_games, damage_ratio=0.01, ledge_grab_penalty=0.01, approaching_factor=0, stalling_penalty=0
    )
    print("Batched Rewards:\n", rewards_batched)
    print("Batched Rewards shape:", rewards_batched.shape)
    
    # The shape should be (2, 2) because:
    # - We have 3 frames of input data, which results in 2 rewards (T-1)
    # - We have 2 batched games (B=2)
    assert rewards_batched.shape == (2, 2), "Expected shape (2, 2) for batched rewards"

    # create another mock singles game with p2 as the opponent
    game3p0 = create_mock_player(action=np.array([0xE, 0x0, 0x0]), percent=np.array([0, 0, 0]), stocks_left=np.array([1, 0, 0]), character=np.array([5, 5, 5]), frames=3)
    game3p1 = create_mock_player(action=np.array([0, 0, 0]), percent=np.array([0, 0, 0]), stocks_left=np.array([0, 0, 0]), character=np.array([0, 0, 0]), frames=3)
    game3p2 = create_mock_player(action=np.array([0, 0, 0]), percent=np.array([0, 0, 30]), stocks_left=np.array([3, 3, 3]), character=np.array([0, 0, 0]), frames=3)
    game3p3 = create_mock_player(action=np.array([0, 0, 0]), percent=np.array([0, 10, 10]), stocks_left=np.array([0, 0, 0]), character=np.array([0, 0, 0]), frames=3, dead=True)

    game3 = Game(p0=game3p0, p1=game3p1, p2=game3p2, p3=game3p3, stage=np.array([0, 0, 0]), randall_phase=np.array([0, 0, 0]), is_teams=np.array([False, False, False]))
    batched_games = batch_games(game, game2, game3)

    rewards_batched = reward.compute_rewards(
        batched_games, damage_ratio=0.01, ledge_grab_penalty=0.01, approaching_factor=0, stalling_penalty=0
    )
    print("Batched Rewards: \n", rewards_batched)
    assert rewards_batched.shape == (2, 3), "Expected shape (2, 3) for batched rewards"

    # create another mock doubles game
    game4p0 = create_mock_player(action=np.array([0xE, 0xE, 0x0]), percent=np.array([0, 0, 0]), stocks_left=np.array([1, 1, 0]), character=np.array([6, 6, 6]), frames=3)
    game4p1 = create_mock_player(action=np.array([0, 0, 0]), percent=np.array([0, 0, 0]), stocks_left=np.array([0, 0, 0]), character=np.array([7, 7, 7]), frames=3)
    game4p2 = create_mock_player(action=np.array([0, 0, 0]), percent=np.array([0, 0, 0]), stocks_left=np.array([4, 4, 4]), character=np.array([0, 0, 0]), frames=3, dead=True)
    game4p3 = create_mock_player(action=np.array([0, 0, 0]), percent=np.array([0, 0, 0]), stocks_left=np.array([4, 4, 4]), character=np.array([0, 0, 0]), frames=3)

    game4 = Game(p0=game4p0, p1=game4p1, p2=game4p2, p3=game4p3, stage=np.array([0, 0, 0]), randall_phase=np.array([0, 0, 0]), is_teams=np.array([True, True, True]))
    game3 = Game(p0=game3p0, p1=game3p1, p2=game3p2, p3=game3p3, stage=np.array([0, 0, 0]), randall_phase=np.array([0, 0, 0]), is_teams=np.array([False, False, False]))
    batched_games = batch_games(game, game2, game3, game4)

    rewards_batched = reward.compute_rewards(
        batched_games, damage_ratio=0.01, ledge_grab_penalty=0.01, approaching_factor=0, stalling_penalty=0
    )
    print("Batched Rewards: \n", rewards_batched)
    assert rewards_batched.shape == (2, 4), "Expected shape (2, 4) for batched rewards"

    expected_rewards = [
        [-0.25,        0.19999999, -4.,          0.        ],
        [-0.33333334,  1.,          0.29999998, -4.,        ]
    ]
    expected_rewards = np.array(expected_rewards)
    np.testing.assert_allclose(expected_rewards, rewards_batched, rtol=1e-5)

# Run the test
compute_rewards_test()
