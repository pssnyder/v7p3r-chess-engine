#!/usr/bin/env python3
"""
Test time management improvements for v4.2
Compare old vs new time allocation strategies
"""

def old_time_allocation(remaining_ms):
    """Old conservative 2% allocation"""
    return max(0.05, remaining_ms / 1000.0 * 0.02)

def new_time_allocation(remaining_ms):
    """New aggressive allocation based on game phase"""
    remaining_seconds = remaining_ms / 1000.0
    
    if remaining_seconds > 60:
        # Early game: use 6-8% of remaining time
        return max(0.5, remaining_seconds * 0.07)
    elif remaining_seconds > 30:
        # Mid game: use 8-10% of remaining time  
        return max(0.3, remaining_seconds * 0.09)
    elif remaining_seconds > 10:
        # Late game: use 12-15% of remaining time
        return max(0.2, remaining_seconds * 0.13)
    else:
        # Critical time: use 20% but minimum 0.1s
        return max(0.1, remaining_seconds * 0.20)

def test_time_scenarios():
    """Test various game time scenarios"""
    
    print("Time Management Comparison (Old vs New)")
    print("=" * 50)
    print("Remaining Time | Old (2%) | New (Adaptive) | Improvement")
    print("-" * 55)
    
    scenarios = [
        120000,  # 2 minutes (120s)
        90000,   # 1.5 minutes  
        60000,   # 1 minute
        45000,   # 45 seconds
        30000,   # 30 seconds
        20000,   # 20 seconds
        10000,   # 10 seconds
        5000,    # 5 seconds
        2000,    # 2 seconds
        1000,    # 1 second
    ]
    
    for ms in scenarios:
        old_time = old_time_allocation(ms)
        new_time = new_time_allocation(ms)
        improvement = new_time / old_time if old_time > 0 else 0
        
        print(f"{ms/1000:>8.0f}s      | {old_time:>6.2f}s  | {new_time:>8.2f}s     | {improvement:>5.1f}x")

def test_120_second_game():
    """Simulate time allocation over a 40-move 120-second game"""
    
    print("\n\n120-Second Game Simulation (40 moves)")
    print("=" * 50)
    print("Move | Remaining | Old Time | New Time | New/Old Ratio")
    print("-" * 55)
    
    remaining_time = 120000  # 120 seconds in ms
    
    for move in range(1, 41):
        old_time = old_time_allocation(remaining_time)
        new_time = new_time_allocation(remaining_time)
        ratio = new_time / old_time if old_time > 0 else 0
        
        print(f"{move:>4} | {remaining_time/1000:>7.1f}s | {old_time:>6.2f}s | {new_time:>6.2f}s | {ratio:>9.1f}x")
        
        # Simulate time usage (assume we use the allocated time)
        remaining_time -= new_time * 1000
        
        if remaining_time <= 0:
            print(f"Game would end at move {move} with new allocation")
            break

if __name__ == "__main__":
    test_time_scenarios()
    test_120_second_game()
