#!/usr/bin/env ruby
#Usage: cat sampleinput.txt | ./generatetoc.rb > sampleoutput.txt

toc = "# Table of Contents\n"
newmd = ""

ARGF.each_line do |line|
  newmd << line
  next if !line.start_with?("#")

  heading = line.gsub("#", "").strip
  href = heading.gsub(" ", "-").downcase

  newmd << "<a name=\"#{href}\"></a>\n"
  toc << "\t" * (line.count("#")-1) + "- [#{heading}](\##{href})\n"

end

puts toc + "\n" + newmd
