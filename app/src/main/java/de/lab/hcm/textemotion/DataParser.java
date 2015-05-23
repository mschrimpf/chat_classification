package de.lab.hcm.textemotion;

/**
 * Created by chi-tai on 05.05.15.

 * Chi-Tai Dang
 * HCM-Lab
 * University of Augsburg
 */

import android.content.Context;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SortedMap;
import java.util.TreeMap;

public class DataParser {

    private static final String className = "DataParser";

    private Context mContext;

    public DataParser(Context context){
        mContext = context;
    }

    public List<Category> loadCategories() throws Exception {

        List<Category> categories = new ArrayList<Category>();

        android.util.Log.d(className, "Loading categories ...");

        InputStream iS = mContext.getAssets().open("StdLIWCCategories.txt");
        BufferedReader reader = new BufferedReader(new InputStreamReader(iS));

        String line;
        while ((line = reader.readLine()) != null) {
            String[] values = line.split("\\t");

            if (values.length < 2)
                continue;
            Category category = new Category();
            category.name = values[1];
            category.index = Integer.parseInt(values[0]);

            categories.add(category);
        }

        reader.close();
        return categories;
    }


    public SortedMap<String, List<Integer>> loadCatMap() throws Exception
    {
        SortedMap<String, List<Integer>> catMap = new TreeMap<String, List<Integer>>();

        android.util.Log.d(className, "Loading category map ...");

        InputStream iS = mContext.getAssets().open("DefaultDictionary.txt");
        BufferedReader reader = new BufferedReader(new InputStreamReader(iS));

        String line;
        while ((line = reader.readLine()) != null) {
            String[] values = line.split(" ");
            String category = values[0].replace("*", ".*"); //Required so that we can use java regex expressions
            //category = category.replace("'", ""); //don't and dont should be seen as one word? might give us some trouble afterwards
            List<Integer> categories = new ArrayList<>();
            for (int i = values.length - 1; !values[i].equals(""); i--) {
                categories.add(Integer.valueOf(values[i]));
            }
            catMap.put(category, categories);
        }

        reader.close();
        return catMap;
    }


    public List<String> loadDataLines(String fn) throws IOException {
        List<String> ret = new ArrayList<String>();

        InputStream iS = mContext.getAssets().open(fn);
        BufferedReader br = new BufferedReader(new InputStreamReader(iS));

        String line;
        while((line = br.readLine()) != null) {
            ret.add(line);
        }

        br.close();
        return ret;
    }

}

