# 🖥️ ZyndAI Agent Web Interface

## 📋 Overview
Simple web interface for the ZyndAI Labelling Agent with drag-and-drop file uploads and command processing.

## 🚀 Quick Start

### **1. Install Dependencies**
```bash
pip install flask requests
```

### **2. Run Interface Server**
```bash
python server.py
```

### **3. Run Agent**
```bash
python labelling_agent.py
```

### **4. Access Interface**
Open browser: http://localhost:8080

## 🎯 Features

### **📁 File Upload**
- **Images**: JPG, PNG, WebP, BMP, TIFF
- **Text**: CSV, JSON, TXT, PDF
- **Drag & Drop**: Simply drag files onto upload area

### **💬 Command Processing**
- **Vision Commands**: 
  - `detect dogs and cats in image.jpg`
  - `segment all animals in image.jpg`
  - `label image.jpg`
- **Text Commands**:
  - `classify data.csv by sentiment`
  - `label reviews.csv as positive/negative/neutral`

### **📊 Results Display**
- **JSON Output**: Formatted responses from ZyndAI agent
- **Error Handling**: Clear error messages
- **Loading States**: Visual feedback during processing

## 🔧 Technical Details

### **🌐 Architecture**
```
Browser (8080) → Flask Server → ZyndAI Agent (5003)
```

### **📁 File Handling**
- Temporary storage in `temp/` directory
- Automatic cleanup after processing
- Secure filename handling

### **🔐 Security**
- Local development only
- No external API exposure
- File type validation

## 🎨 Interface Preview

The interface provides:
- **Clean, modern design** with gradient background
- **Responsive layout** for different screen sizes
- **Visual feedback** for all operations
- **Real-time processing** status updates

## 📝 Usage Examples

### **Image Processing:**
1. Upload `image-test.jpg`
2. Enter command: `detect dogs and cats in image-test.jpg`
3. Click "🚀 Process"
4. View JSON results with detected objects

### **Text Processing:**
1. Upload `reviews_unlabelled.csv`
2. Enter command: `classify reviews_unlabelled.csv by sentiment`
3. Click "🚀 Process"
4. View sentiment analysis results

## 🔗 Integration

The interface acts as a **proxy** between users and the ZyndAI agent, providing:
- **Simplified user experience**
- **File management capabilities**
- **Command execution**
- **Result visualization**

**Perfect for testing and demonstrating agent capabilities!** ✨
