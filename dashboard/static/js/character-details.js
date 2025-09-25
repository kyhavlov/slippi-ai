// References to DOM elements
const characterRows = document.querySelectorAll('.character-row');
const characterDetails = document.getElementById('character-details');
const characterNameHeader = document.getElementById('selected-character-name');
const teammateStatsTable = document.getElementById('teammate-stats').getElementsByTagName('tbody')[0];
const playerStatsTable = document.getElementById('player-stats').getElementsByTagName('tbody')[0];
const opponentStatsTable = document.getElementById('opponent-stats').getElementsByTagName('tbody')[0];

// Function to show character details
function showCharacterDetails(character) {
    // Update header
    characterNameHeader.textContent = `${character} Details`;
    
    // Clear previous data
    teammateStatsTable.innerHTML = '';
    playerStatsTable.innerHTML = '';
    opponentStatsTable.innerHTML = '';
    
    // Add teammate stats - sorted by diff
    if (charTeammateStats[character]) {
        const sortedTeammates = Object.entries(charTeammateStats[character])
            .sort((a, b) => b[1].diff - a[1].diff);
        
        sortedTeammates.forEach(([teammate, stats]) => {
            const row = teammateStatsTable.insertRow();
            row.innerHTML = `
                <td>${teammate}</td>
                <td>${stats.rate}%</td>
                <td class="${stats.diff > 0 ? 'positive-diff' : stats.diff < 0 ? 'negative-diff' : ''}">${stats.diff > 0 ? '+' : ''}${stats.diff}%</td>
                <td>${stats.wins}</td>
                <td>${stats.games}</td>
            `;
        });
    }
    
    // Add player stats - sorted by diff
    if (charPlayerStats[character]) {
        const sortedPlayers = Object.entries(charPlayerStats[character])
            .sort((a, b) => b[1].diff - a[1].diff);
        
        sortedPlayers.forEach(([player, stats]) => {
            const row = playerStatsTable.insertRow();
            row.innerHTML = `
                <td>${player}</td>
                <td>${stats.rate}%</td>
                <td class="${stats.diff > 0 ? 'positive-diff' : stats.diff < 0 ? 'negative-diff' : ''}">${stats.diff > 0 ? '+' : ''}${stats.diff}%</td>
                <td>${stats.wins}</td>
                <td>${stats.games}</td>
            `;
        });
    }
    
    // Add opponent stats - sorted by diff
    if (charOpponentStats[character]) {
        const sortedOpponents = Object.entries(charOpponentStats[character])
            .sort((a, b) => b[1].diff - a[1].diff);
        
        sortedOpponents.forEach(([opponent, stats]) => {
            const row = opponentStatsTable.insertRow();
            row.innerHTML = `
                <td>${opponent}</td>
                <td>${stats.rate}%</td>
                <td class="${stats.diff > 0 ? 'positive-diff' : stats.diff < 0 ? 'negative-diff' : ''}">${stats.diff > 0 ? '+' : ''}${stats.diff}%</td>
                <td>${stats.wins}</td>
                <td>${stats.games}</td>
            `;
        });
    }
    
    // Show the details section
    characterDetails.style.display = 'block';
}

// Add click event listeners to character rows
characterRows.forEach(row => {
    row.addEventListener('click', function() {
        // Remove selected class from all rows
        characterRows.forEach(r => r.classList.remove('selected-character'));
        // Add selected class to clicked row
        this.classList.add('selected-character');
        
        // Get character name and show details
        const character = this.dataset.character;
        showCharacterDetails(character);
    });
}); 