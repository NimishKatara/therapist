import 'dotenv/config';
import express from 'express';
import fetch from 'node-fetch';

const app = express();
app.use(express.json());

const API_KEY = process.env.GEMINI_API_KEY; // Secure API Key
const API_URL = `https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=${API_KEY}`;

app.post('/ask', async (req, res) => {
    try {
        const { prompt } = req.body;
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                contents: [{ parts: [{ text: prompt }] }]
            })
        });

        const data = await response.json();
        res.json(data);
    } catch (error) {
        res.status(500).json({ error: "Something went wrong" });
    }
});

app.listen(3000, () => console.log('Server running on port 3000'));
