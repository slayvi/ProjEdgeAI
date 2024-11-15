package com.example.testcv2;

import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import android.widget.TextView;
import android.widget.Toast;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.HashMap;
import java.util.Map;

import android.Manifest;
import android.view.View;
import android.content.res.ColorStateList;
import android.graphics.Color;

import org.tensorflow.lite.Interpreter;

public class MainActivity extends AppCompatActivity {

    private static final int CAMERA_PERMISSION_CODE = 100;
    private static final int STORAGE_PERMISSION_CODE = 101;
    private static final int CAMERA_REQUEST = 1888;
    private static final int GALLERY_REQUEST = 1889;

    private final int colorActive = Color.parseColor("#3b6e96");
    private final int colorInactive = Color.parseColor("#7c9eb9");

    private ImageView imageView;
    private TextView textLatency;

    private Bitmap selectedBitmap;

    private Interpreter ageGenderModel, ageGenderModelQuantized, ageGenderModelPruned;
    private Interpreter emotionModel, emotionModelQuantized, emotionModelPruned;
    private Interpreter currentAgeGenderModel;
    private Interpreter currentEmotionModel;

    private Button buttonNormal;
    private Button buttonQuantized;
    private Button buttonPruned;

    private ByteBuffer ageGenderBuffer, emotionBuffer;
    private String currentMode = "Normal"; // Track the current mode

    private CascadeClassifier faceCascade;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageView = findViewById(R.id.imageView);
        textLatency = findViewById(R.id.text_latency);

        Button buttonCamera = findViewById(R.id.button_camera);
        Button buttonSelect = findViewById(R.id.button_select);
        Button buttonPredict = findViewById(R.id.button_predict);

        buttonNormal = findViewById(R.id.button_normal);
        buttonQuantized = findViewById(R.id.button_quantized);
        buttonPruned = findViewById(R.id.button_pruned);


        // Initialize OpenCV
        if (OpenCVLoader.initDebug()) {
            Log.i("OpenCV", "OpenCV successfully loaded.");
        }

        loadHaarCascade();
        loadModels();

        // Set colors
        buttonNormal.setBackgroundTintList(ColorStateList.valueOf(colorInactive));
        buttonQuantized.setBackgroundTintList(ColorStateList.valueOf(colorInactive));
        buttonPruned.setBackgroundTintList(ColorStateList.valueOf(colorInactive));

        // Set initial mode to Normal
        setActiveModel(buttonNormal, "Normal");
        Log.e("MainActivity", currentMode);

        // Set click listeners
        buttonNormal.setOnClickListener(v -> setActiveModel(buttonNormal, "Normal"));
        buttonQuantized.setOnClickListener(v -> setActiveModel(buttonQuantized, "Quantized"));
        buttonPruned.setOnClickListener(v -> setActiveModel(buttonPruned, "Pruned"));

        buttonCamera.setOnClickListener(v -> openCamera());
        buttonSelect.setOnClickListener(v -> openGallery());

        buttonPredict.setOnClickListener(v -> {
            if (selectedBitmap != null) {
                faceDetection(selectedBitmap);
            } else {
                Toast.makeText(MainActivity.this, "Please select an image", Toast.LENGTH_SHORT).show();
            }
        });



    }

    private void openCamera() {
        if (ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(MainActivity.this, new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION_CODE);
        } else {
            Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
            startActivityForResult(cameraIntent, CAMERA_REQUEST);
        }
    }

    private void openGallery() {
        Intent galleryIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(galleryIntent, GALLERY_REQUEST);
    }

    private void setActiveModel(Button activeButton, String mode) {
        currentMode = mode;

        // Loading models
        switch (mode) {
            case "Normal":
                currentAgeGenderModel = ageGenderModel;
                currentEmotionModel = emotionModel;
                break;

            case "Quantized":
                currentAgeGenderModel = ageGenderModelQuantized;
                currentEmotionModel = emotionModelQuantized;
                break;

            case "Pruned":
                currentAgeGenderModel = ageGenderModelPruned;
                currentEmotionModel = emotionModelPruned;
                break;
        }

        // Reset button colors
        buttonNormal.setBackgroundTintList(ColorStateList.valueOf(colorInactive));
        buttonQuantized.setBackgroundTintList(ColorStateList.valueOf(colorInactive));
        buttonPruned.setBackgroundTintList(ColorStateList.valueOf(colorInactive));

        // change color of active button
        activeButton.setBackgroundTintList(ColorStateList.valueOf(colorActive));

        Log.e("MainActivity", "Mode changed to " + currentMode);

    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK) {
            try {
                if (requestCode == CAMERA_REQUEST && data != null) {
                    selectedBitmap = (Bitmap) data.getExtras().get("data");
                } else if (requestCode == GALLERY_REQUEST && data != null) {
                    Uri selectedImage = data.getData();
                    selectedBitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), selectedImage);
                }
                imageView.setImageBitmap(selectedBitmap);

                // Hide "Please select Image" text
                TextView placeholderText = findViewById(R.id.placeholderText);
                placeholderText.setVisibility(View.GONE);

            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }



    private void loadHaarCascade() {
        try {
            InputStream is = getAssets().open("haarcascade_frontalface_default.xml");
            File cascadeDir = getDir("cascade", MODE_PRIVATE);
            File cascadeFile = new File(cascadeDir, "haarcascade_frontalface_default.xml");

            FileOutputStream fos = new FileOutputStream(cascadeFile);
            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                fos.write(buffer, 0, bytesRead);
            }
            is.close();
            fos.close();

            faceCascade = new CascadeClassifier(cascadeFile.getAbsolutePath());
            if (faceCascade.empty()) {
                Log.e("MainActivity", "Failed to load cascade classifier.");
                faceCascade = null;
            } else {
                Log.i("MainActivity", "Loaded Haar Cascade classifier.");
            }
            cascadeDir.delete();
        } catch (IOException e) {
            Log.e("MainActivity", "Error loading cascade", e);
        }
    }


    private void loadModels() {
        try {
            ageGenderModel = new Interpreter(loadModelFile("2classes_final_age_gender_model.tflite"));
            ageGenderModelQuantized = new Interpreter(loadModelFile("2classes_final_age_gender_model_quant.tflite"));
            ageGenderModelPruned = new Interpreter((loadModelFile("2classes_final_pruned_quant_agegender.tflite")));

            emotionModel = new Interpreter(loadModelFile("TFN3_final_tf12_emotion_model.tflite"));
            emotionModelQuantized = new Interpreter(loadModelFile("TFN3_final_tf12_emotion_model_quantized.tflite"));
            emotionModelPruned = new Interpreter(loadModelFile("TFN3_final_emotion_pruned_quantized_model.tflite"));

            currentAgeGenderModel = ageGenderModel;
            currentEmotionModel = emotionModel;


            ageGenderBuffer = ByteBuffer.allocateDirect(4 * 128 * 128).order(ByteOrder.nativeOrder());
            emotionBuffer = ByteBuffer.allocateDirect(4 * 48 * 48).order(ByteOrder.nativeOrder());

        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    private Bitmap faceDetection(Bitmap bitmap) {

        TextView classificationResults = findViewById(R.id.classificationResults);
        TextView noFaceDetectedText = findViewById(R.id.noFaceDetectedText);

        // Preprocessing for cv2 and grayscaling
        Mat matBitmap = new Mat();
        Utils.bitmapToMat(bitmap, matBitmap);

        Mat grayMat = new Mat();
        Imgproc.cvtColor(matBitmap, grayMat, Imgproc.COLOR_BGR2GRAY);

        // detect face
        MatOfRect faces = new MatOfRect();
        faceCascade.detectMultiScale(grayMat, faces);


        if (faces.empty()) {
            // if no face is found show text
            noFaceDetectedText.setVisibility(View.VISIBLE);
            classificationResults.setText("No face detected");

        } else {

            // Looking for the largest face (area)
            Rect largestFace = null;
            double maxArea = 0;
            for (Rect face : faces.toArray()) {
                double area = face.width * face.height;
                if (area > maxArea) {
                    maxArea = area;
                    largestFace = face;
                }
            }


            // Process only the largest face
            if (largestFace != null) {
                Bitmap faceBitmap = cropFace(bitmap, largestFace);

                // measure latency and get predictions
                long startTime = System.nanoTime();
                String ageGenderPrediction = predictAgeGender(faceBitmap);
                String emotionPrediction = predictEmotion(faceBitmap);
                long endTime = System.nanoTime();

                // Calculating and showing latency
                long latency = (endTime - startTime) / 1_000_000;
                textLatency.setText("Latency: " + latency + " ms (" + currentMode + " Mode)");

                // Showing predictions
                String resultText = ageGenderPrediction + "\n" + emotionPrediction;
                classificationResults.setText(resultText);

                // plotting rectangle to face
                Imgproc.rectangle(matBitmap, largestFace.tl(), largestFace.br(), new Scalar(0, 0, 0), 3);

            }
        }


        // Convert the Mat back to Bitmap to show it correctly on the screen
        Bitmap resultBitmap = Bitmap.createBitmap(matBitmap.width(), matBitmap.height(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(matBitmap, resultBitmap);
        grayMat.release();
        matBitmap.release();
        imageView.setImageBitmap(resultBitmap);

        return resultBitmap;
    }


    private String predictAgeGender(Bitmap faceBitmap) {
        // Preprocessing
        preprocessImage(faceBitmap, 128, ageGenderBuffer);

        // Output Arrays
        float[][] outputAge = new float[1][1];
        float[][] outputGender = new float[1][1];

        // Use a Map to store multiple outputs
        Object[] inputs = {ageGenderBuffer};
        Map<Integer, Object> outputs = new HashMap<>();

        outputs.put(0, outputAge);
        outputs.put(1, outputGender);

        // Run the model
        currentAgeGenderModel.runForMultipleInputsOutputs(inputs, outputs);

        // Process prediction
        float genProb = outputGender[0][0];
        String gender = genProb > 0.5 ? "Female" : "Male";
        float genderProb = genProb > 0.5 ? genProb : (1 - genProb);

        float ageProb = outputAge[0][0];
        String age = ageProb > 0.5 ? "Elderly" : "Adult";
        float agePrintProb = ageProb > 0.5 ? ageProb : (1 - ageProb);


        // Return text with prediction results
        return "Gender: " + gender + " (" + String.format("%.2f", genderProb*100) + " %)\n " +
                "Age: " + age + " (" + String.format("%.2f", agePrintProb*100) + " %)";
    }



    private String predictEmotion(Bitmap faceBitmap) {
        // Preprocessing
        preprocessImage(faceBitmap, 48, emotionBuffer);

        // Output Array
        float[][] outputEmotion = new float[1][7];

        // Run the model
        currentEmotionModel.run(emotionBuffer, outputEmotion);

        // Processing prediction result and returning text with them
        int maxIndex = getMaxIndex(outputEmotion[0]);
        String[] emotionCategories = {"Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"};

        return "Emotion: " + emotionCategories[maxIndex] + " (" + String.format("%.2f", outputEmotion[0][maxIndex]*100) + " %)";
    }


    private void preprocessImage(Bitmap bitmap, int size, ByteBuffer buffer) {
        // Preprocessing to change shape, color of face to model inputs, normalize

        Bitmap resized = Bitmap.createScaledBitmap(bitmap, size, size, false);
        Mat mat = new Mat();
        Utils.bitmapToMat(resized, mat);

        Mat grayMat = new Mat();
        Imgproc.cvtColor(mat, grayMat, Imgproc.COLOR_BGR2GRAY); // grayscale
        grayMat.convertTo(grayMat, CvType.CV_32F, 1.0 / 255.0); // normalize

        float[] floatValues = new float[size * size];
        grayMat.get(0, 0, floatValues);
        buffer.rewind(); // Reset buffer position
        for (float val : floatValues) {
            buffer.putFloat(val);
        }
    }

    private int getMaxIndex(float[] arr) {
        int maxIndex = 0;
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > arr[maxIndex]) maxIndex = i;
        }
        return maxIndex;
    }

    private MappedByteBuffer loadModelFile(String modelFileName) throws IOException {
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd(modelFileName);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, fileDescriptor.getStartOffset(), fileDescriptor.getDeclaredLength());
    }

    private Bitmap cropFace(Bitmap bitmap, Rect face) {
        return Bitmap.createBitmap(bitmap, face.x, face.y, face.width, face.height);
    }
}
