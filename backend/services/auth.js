const jwt = require("jsonwebtoken");
const SECRET_KEY = process.env.JWT_SECRET || "4EC555C92F4AF519F1215D6B598C6";


function generateJwtToken(user) {
    try {
        return jwt.sign({
            _id: user._id
        }, SECRET_KEY, {expiresIn: "7d"});
    } catch(error) {
        return undefined;
    }
}

function verifyJwtToken(token) {
    try {
        const decoded = jwt.verify(token, SECRET_KEY);
        return decoded._id;
    } catch (err) {
        return null;
    }
}

module.exports = {
    generateJwtToken,
    verifyJwtToken
}