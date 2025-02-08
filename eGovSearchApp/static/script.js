document.getElementById('search-btn').addEventListener('click', async () => {
    const question = document.getElementById('question').value;
    const resultsContainer = document.getElementById('results');
    resultsContainer.innerHTML = ''; // 検索結果をリセット

    if (!question.trim()) {
        alert('質問を入力してください。');
        return;
    }

    console.log(`検索クエリ: ${question}`);

    try {
        const response = await fetch('http://localhost:8000/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question, top_n: 3 }), // top_n を適切に設定
        });

        console.log(`API ステータスコード: ${response.status}`);

        if (!response.ok) {
            throw new Error(`エラー: ${response.statusText}`);
        }

        const data = await response.json();
        console.log('API 応答データ:', data);

        const results = data.results;

        if (results.length === 0) {
            resultsContainer.innerHTML = '<p>結果が見つかりませんでした。</p>';
            return;
        }

        results.forEach(result => {
            console.log('検索結果アイテム:', result);
            const resultDiv = document.createElement('div');
            resultDiv.className = 'result-item';

            const law_num = document.createElement('h3');
            law_num.textContent = result.law_num;

            const law_title = document.createElement('p');
            law_title.textContent = result.law_title;

            const law_text = document.createElement('p');
            law_text.textContent = result.law_text;

            resultDiv.appendChild(law_num);
            resultDiv.appendChild(law_title);
            resultDiv.appendChild(law_text);
            resultsContainer.appendChild(resultDiv);
        });
    } catch (error) {
        console.error('エラー:', error);
        resultsContainer.innerHTML = `<p>エラーが発生しました: ${error.message}</p>`;
    }
});

