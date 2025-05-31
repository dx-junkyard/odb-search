document.getElementById('search-btn').addEventListener('click', async () => {
    const question = document.getElementById('question').value;
    const resultsContainer = document.getElementById('results');
    resultsContainer.innerHTML = ''; // 検索結果をリセット

    if (!question.trim()) {
        alert('質問を入力してください。');
        return;
    }

    try {
        const response = await fetch('http://localhost:8000/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question, top_n: 10 }), // top_n を適切に設定
        });

        if (!response.ok) {
            throw new Error(`エラー: ${response.statusText}`);
        }

        const data = await response.json();
        const results = data.results;

        if (results.length === 0) {
            resultsContainer.innerHTML = '<p>結果が見つかりませんでした。</p>';
            return;
        }

        results.forEach(result => {
            const resultDiv = document.createElement('div');
            resultDiv.className = 'result-item';

            const title = document.createElement('h3');
            title.textContent = result.formal_name;

            const overview = document.createElement('p');
            overview.textContent = result.overview;

            const link = document.createElement('a');
            link.href = result.url;
            link.textContent = '詳細を見る';
            link.target = '_blank';

            resultDiv.appendChild(title);
            resultDiv.appendChild(overview);
            resultDiv.appendChild(link);
            resultsContainer.appendChild(resultDiv);
        });
    } catch (error) {
        console.error('エラー:', error);
        resultsContainer.innerHTML = `<p>エラーが発生しました: ${error.message}</p>`;
    }
});

