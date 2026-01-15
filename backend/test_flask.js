const axios = require("axios");

const FLASK_URL = "http://127.0.0.1:5000";

async function testFlask() {
    console.log("Testing Flask server...\n");

    try {
        // Test health check
        console.log("1. Testing health check endpoint...");
        const healthResponse = await axios.get(`${FLASK_URL}/`);
        console.log("✓ Health check passed:", healthResponse.data);

        // Test if predict endpoint exists
        console.log("\n2. Testing predict endpoint (without image)...");
        try {
            await axios.post(`${FLASK_URL}/predict`);
        } catch (error) {
            if (error.response && error.response.status === 400) {
                console.log("✓ Predict endpoint exists (expected 400 error without image)");
            } else {
                throw error;
            }
        }

        console.log("\n✓✓✓ Flask server is running correctly! ✓✓✓");
        console.log("You can now use your application.");

    } catch (error) {
        console.error("\n✗✗✗ Flask server test failed ✗✗✗");

        if (error.code === 'ECONNREFUSED') {
            console.error("\nProblem: Flask server is NOT running");
            console.error("Solution: Run 'python main.py' in another terminal");
        } else if (error.code === 'ETIMEDOUT') {
            console.error("\nProblem: Connection timeout");
            console.error("Solution: Check if Flask is running on the correct port");
        } else {
            console.error("\nError:", error.message);
        }
    }
}

testFlask();
